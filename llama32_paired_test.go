package llama

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/itsmontoya/llama/internal/tokenizer"
)

type pairedLane struct {
	PrefillNS   int64
	TTFTNS      int64
	DecodeNS    []int64
	DecodeEndNS []int64
	WallNS      int64
	FirstNS     int64
	Winners     []int64
	RejectedNS  []int64
	Rows        [][]float32 `json:"-"`
	Err         error       `json:"-"`
}

type pairedPhase struct {
	CacheNS       int64
	WallNS        int64
	AllocBytes    uint64
	Allocations   uint64
	Lanes         []pairedLane
	FrozenErrors  []float64
	MaxFrozenFrac float64
	NearTieMargin []float64
}

func TestLlama32PairedCampaign(t *testing.T) {
	var (
		backend = os.Getenv("LLAMA_PAIRED_BACKEND")
		dir     = os.Getenv("LLAMA_32_1B_MODEL_DIR")
		prefix  = os.Getenv("LLAMA_PAIRED_OUTPUT")
		work    = os.Getenv("LLAMA_PAIRED_CASE")
	)
	if backend == "" {
		t.Skip("run benchmarks/metal/campaign.py for the paired fixture experiment")
	}
	// The controller observes the start and first-token markers on this pipe.
	fmt.Println("LLAMA_READY")
	if dir == "" || prefix == "" {
		t.Fatal("paired binary requires the verified fixture and an artifact prefix")
	}
	var fixture struct {
		Revision   string `json:"revision"`
		Generation struct {
			PromptIDs []int64                   `json:"prompt_ids"`
			SuffixIDs []int64                   `json:"suffix_ids"`
			Steps     []llama32LogitObservation `json:"steps"`
		} `json:"generation"`
	}
	var err error
	if err = readHFJSON(filepath.Join("testdata", "llama32_reference.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Revision != llama32Revision || len(fixture.Generation.PromptIDs) != 6 || len(fixture.Generation.SuffixIDs) != 8 || len(fixture.Generation.Steps) != 8 {
		t.Fatal("invalid pinned generation fixture")
	}
	var loadStart = time.Now()
	var decoders = make([]*Decoder, 1, 2)
	defer func() {
		for _, decoder := range decoders {
			if decoder != nil {
				_ = decoder.Close()
			}
		}
	}()
	var loads []int64
	var loadOne = func() (decoder *Decoder, generation HFGenerationConfig) {
		var start = time.Now()
		switch backend {
		case "cpu":
			decoder, generation, err = LoadHFModel(dir, 512)
		case "metal":
			decoder, generation, err = LoadHFModelMetalAttention(t.Context(), dir, 512)
		default:
			t.Fatal("unknown paired backend", backend)
		}
		loads = append(loads, time.Since(start).Nanoseconds())
		if err != nil {
			t.Fatal(err)
		}
		return decoder, generation
	}
	var generation HFGenerationConfig
	decoders[0], generation = loadOne()
	if work == "independent" {
		var second, _ = loadOne()
		decoders = append(decoders, second)
	}
	var modelLoadNS = time.Since(loadStart).Nanoseconds()
	var tokenizeStart = time.Now()
	var vocab *tokenizer.Llama3Tokenizer
	vocab, err = tokenizer.LoadLlama3(dir, decoders[0].Config().VocabSize, generation.BOSID, generation.StopIDs)
	if err != nil {
		t.Fatal(err)
	}
	var prompt []int64
	prompt, err = vocab.Encode(llama32ShortPrompt)
	if err != nil || !slices.Equal(prompt, fixture.Generation.PromptIDs) {
		t.Fatalf("prompt IDs %v, error %v", prompt, err)
	}
	var (
		want       = fixture.Generation.SuffixIDs
		frozen     = fixture.Generation.Steps
		laneCount  = 1
		tailLength int
		hostDigest string
	)
	switch work {
	case "short", "recovery":
	case "sustained", "shared", "independent":
		want = []int64{
			12366, 13, 1102, 374, 279, 1455, 95551, 3363, 304, 9822, 323, 279,
			3224, 753, 6864, 323, 7928, 3363, 13, 12366, 374, 279, 1455, 95551,
			16036, 3158, 304, 279, 7665, 9323, 323, 279, 1917,
		}
		hostDigest = "b72d983e8b5c45f83ad858b52f1d27b87a7ad8118be6c5d315db2ea6fc6fd84a"
		if work != "sustained" {
			laneCount = 2
		}
	case "concurrent-prefill":
		laneCount = 2
		want = want[:1]
		hostDigest = "f2e070e41d1eac82f0159284cfdd0d070aa06ef703c833fa6fb781a5b7b487c1"
	case "27", "28", "29":
		tailLength, err = strconv.Atoi(work)
		if err != nil {
			t.Fatal(err)
		}
		prompt, err = vocab.Encode(llama32LongPrompt)
		if err != nil || len(prompt) != 29 {
			t.Fatalf("long prompt IDs %v, error %v", prompt, err)
		}
		prompt = prompt[:tailLength]
		want = []int64{[]int64{9822, 374, 12366}[tailLength-27]}
		hostDigest = []string{
			"0b087b0839da36ec06a3a8f777d4c3c944ccc7439107fcad39b6ad0239ed9afc",
			"cc93319c3c0701fd9cd3fdc2080afe43d2f82ba7cd236a698d19518836854e8c",
			"6ffb4dc93003480aebdd51e9f1fee47ae3836eea9fb35ce2508f5d618a9871a5",
		}[tailLength-27]
		frozen = nil
	default:
		t.Fatal("unknown paired workload", work)
	}
	var (
		tokenizerNS = time.Since(tokenizeStart).Nanoseconds()
		capacity    = int64(len(prompt) + len(want) - 1)
		modelBytes  int64
		kvBytes     int64
	)
	for _, decoder := range decoders {
		if decoder.metal == nil {
			continue
		}
		for _, tensor := range decoder.metal.weights {
			var bytes = int64(4)
			for _, dimension := range tensor.Shape() {
				bytes *= dimension
			}
			modelBytes += bytes
		}
	}
	if backend == "metal" {
		var config = decoders[0].Config()
		kvBytes = 16 * config.Layers * capacity * config.KVHeads * (config.Dim / config.Heads) * int64(laneCount)
	}
	var profile *os.File
	if os.Getenv("LLAMA_PAIRED_PROFILE") == "1" {
		profile, err = os.Create(prefix + ".cpu.pprof")
		if err != nil {
			t.Fatal(err)
		}
		if err = pprof.StartCPUProfile(profile); err != nil {
			t.Fatal(err)
		}
	}
	var results [2]pairedPhase
	for index := range results {
		results[index] = runPairedPhase(t, decoders, laneCount, capacity, prompt, want, index == 0, work == "recovery", loadStart)
	}
	if profile != nil {
		pprof.StopCPUProfile()
		if err = profile.Close(); err != nil {
			t.Fatal(err)
		}
	}
	for phase := range results {
		var result = &results[phase]
		for i, lane := range result.Lanes {
			for step, row := range lane.Rows {
				if i > 0 && !slices.Equal(row, result.Lanes[0].Rows[step]) {
					t.Fatalf("phase %d position %d lanes disagree", phase, len(prompt)+step-1)
				}
				if phase > 0 && !slices.Equal(row, results[0].Lanes[i].Rows[step]) {
					t.Fatalf("position %d changed between first and warm run", len(prompt)+step-1)
				}
				if step >= len(frozen) {
					continue
				}
				var label = fmt.Sprintf("phase %d lane %d", phase, i)
				var observation = frozen[step]
				var input = append(slices.Clone(prompt), want[:step]...)
				if !slices.Equal(observation.InputIDs, input) {
					t.Fatalf("position %d reference prefix mismatch", observation.Position)
				}
				var _, fraction = compareLlama32Row(t, label, input, row, observation)
				result.MaxFrozenFrac = max(result.MaxFrozenFrac, fraction.fraction)
				for _, sample := range observation.Samples {
					result.FrozenErrors = append(result.FrozenErrors, math.Abs(float64(row[sample.ID])-sample.Logit))
				}
				if observation.Position == 11 {
					result.NearTieMargin = append(result.NearTieMargin, float64(row[observation.Top[0].ID])-float64(row[observation.Top[1].ID]))
				}
			}
			if backend == "cpu" && hostDigest != "" {
				var got = fmt.Sprintf("%x", llama32LogitDigest(lane.Rows[len(lane.Rows)-1]))
				if got != hostDigest {
					t.Fatalf("phase %d lane %d final CPU digest %s, want %s", phase, i, got, hostDigest)
				}
			}
		}
	}
	writePairedRows(t, prefix+".f32", &results)
	// Remove captured result rows before measuring retained runtime memory.
	for phase := range results {
		for lane := range results[phase].Lanes {
			results[phase].Lanes[lane].Rows = nil
		}
	}
	runtime.GC()
	var liveHeap runtime.MemStats
	runtime.ReadMemStats(&liveHeap)
	var retainedRSS = pairedRSS(t)
	if profile != nil {
		var heap *os.File
		heap, err = os.Create(prefix + ".heap.pprof")
		if err != nil {
			t.Fatal(err)
		}
		if err = pprof.WriteHeapProfile(heap); err != nil {
			t.Fatal(err)
		}
		if err = heap.Close(); err != nil {
			t.Fatal(err)
		}
	}
	var closeStart = time.Now()
	for i := range decoders {
		if err = decoders[i].Close(); err != nil {
			t.Fatal(err)
		}
		decoders[i] = nil
	}
	var closeNS = time.Since(closeStart).Nanoseconds()
	runtime.GC()
	var closedHeap runtime.MemStats
	runtime.ReadMemStats(&closedHeap)
	var record = map[string]any{
		"backend": backend, "case": work, "prompt_ids": prompt, "capacity": capacity,
		"loads_ns": loads, "model_load_ns": modelLoadNS, "tokenizer_ns": tokenizerNS,
		"phases": results, "model_bytes": modelBytes, "kv_bytes": kvBytes,
		"retained_rss_kib": retainedRSS, "retained_heap_bytes": liveHeap.HeapAlloc,
		"closed_rss_kib": pairedRSS(t), "closed_heap_bytes": closedHeap.HeapAlloc, "close_ns": closeNS,
		"go": runtime.Version(), "gomaxprocs": runtime.GOMAXPROCS(0),
	}
	var data []byte
	data, err = json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("LLAMA_RESULT %s\n", data)
}

func runPairedPhase(t *testing.T, decoders []*Decoder, laneCount int, capacity int64, prompt, want []int64, cold, recoverStep bool, loadStart time.Time) (out pairedPhase) {
	t.Helper()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	var sessions = make([]*Session, laneCount)
	var start = time.Now()
	for i := range sessions {
		var err error
		sessions[i], err = decoders[i%len(decoders)].NewSession(capacity)
		if err != nil {
			t.Fatal(err)
		}
	}
	out.CacheNS = time.Since(start).Nanoseconds()
	out.Lanes = make([]pairedLane, laneCount)
	var ready, finished sync.WaitGroup
	var barrier = make(chan struct{})
	var first sync.Once
	var begin time.Time
	ready.Add(laneCount)
	finished.Add(laneCount)
	for lane, session := range sessions {
		go func() {
			defer finished.Done()
			ready.Done()
			<-barrier
			out.Lanes[lane] = runPairedLane(session, prompt, want, recoverStep, begin, func(ts time.Time, id int64) {
				if cold {
					first.Do(func() {
						fmt.Printf("LLAMA_FIRST %d %d\n", ts.Sub(loadStart).Nanoseconds(), id)
					})
				}
			})
		}()
	}
	ready.Wait()
	begin = time.Now()
	close(barrier)
	finished.Wait()
	out.WallNS = time.Since(begin).Nanoseconds()
	runtime.ReadMemStats(&after)
	out.AllocBytes = after.TotalAlloc - before.TotalAlloc
	out.Allocations = after.Mallocs - before.Mallocs
	for i, session := range sessions {
		if err := out.Lanes[i].Err; err != nil {
			t.Fatal(err)
		}
		if err := session.Close(); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

func runPairedLane(session *Session, prompt, want []int64, recovery bool, barrier time.Time, first func(time.Time, int64)) (out pairedLane) {
	out.Rows = make([][]float32, 0, len(want))
	out.Winners = make([]int64, 0, len(want))
	out.DecodeNS = make([]int64, 0, len(want)-1)
	out.DecodeEndNS = make([]int64, 0, len(want)-1)
	var start = time.Now()
	var row, err = session.Prefill(prompt)
	out.PrefillNS = time.Since(start).Nanoseconds()
	for step, expected := range want {
		if err != nil {
			out.Err = err
			return out
		}
		var winner, selectErr = greedyToken(row)
		if selectErr != nil || winner != expected {
			out.Err = fmt.Errorf("position %d winner %d, want %d, error %v", session.Position()-1, winner, expected, selectErr)
			return out
		}
		if step == 0 {
			var now = time.Now()
			out.TTFTNS = now.Sub(start).Nanoseconds()
			out.FirstNS = now.Sub(barrier).Nanoseconds()
			first(now, winner)
		} else {
			out.DecodeEndNS = append(out.DecodeEndNS, time.Since(barrier).Nanoseconds())
		}
		out.Rows = append(out.Rows, row)
		out.Winners = append(out.Winners, winner)
		if step+1 == len(want) {
			break
		}
		if recovery && step == 1 {
			var position = session.Position()
			var rejected = time.Now()
			var invalid, invalidErr = session.Decode(-1)
			out.RejectedNS = append(out.RejectedNS, time.Since(rejected).Nanoseconds())
			if invalid != nil || invalidErr == nil || session.Position() != position {
				out.Err = fmt.Errorf("invalid token published position %d, error %v", session.Position(), invalidErr)
				return out
			}
			var canceled, cancel = context.WithCancel(context.Background())
			cancel()
			rejected = time.Now()
			var canceledRow, canceledErr = session.DecodeContext(canceled, expected)
			out.RejectedNS = append(out.RejectedNS, time.Since(rejected).Nanoseconds())
			if canceledRow != nil || !errors.Is(canceledErr, context.Canceled) || session.Position() != position {
				out.Err = fmt.Errorf("cancellation published position %d, error %v", session.Position(), canceledErr)
				return out
			}
		}
		var decodeStart = time.Now()
		row, err = session.Decode(winner)
		out.DecodeNS = append(out.DecodeNS, time.Since(decodeStart).Nanoseconds())
	}
	out.WallNS = time.Since(start).Nanoseconds()
	if session.Position() != int64(len(prompt)+len(want)-1) {
		out.Err = fmt.Errorf("unexpected committed position %d", session.Position())
	}
	return out
}

func writePairedRows(t *testing.T, path string, phases *[2]pairedPhase) {
	t.Helper()
	var file, err = os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var writer = bufio.NewWriterSize(file, 1<<20)
	var data = make([]byte, 128256*4)
	for _, phase := range phases {
		for _, lane := range phase.Lanes {
			for _, row := range lane.Rows {
				if len(row)*4 != len(data) {
					t.Fatalf("unexpected vocabulary width %d", len(row))
				}
				for i, value := range row {
					binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(value))
				}
				if _, err = writer.Write(data); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	if err = writer.Flush(); err != nil {
		t.Fatal(err)
	}
	if err = file.Close(); err != nil {
		t.Fatal(err)
	}
}

func pairedRSS(t *testing.T) (rss int64) {
	t.Helper()
	var data, err = exec.Command("/bin/ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		t.Fatal(err)
	}
	rss, err = strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	return rss
}
