package kmeans

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"
)

type point struct {
	X, Y float64
}

func sqdist(a, b point) float64 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	return dx*dx + dy*dy
}

func mean(points []point) point {
	var s point
	for _, p := range points {
		s.X += p.X
		s.Y += p.Y
	}
	n := float64(len(points))
	return point{s.X / n, s.Y / n}
}

func fixed(centroids ...point) Initializer[point] {
	return func(_ []point, k int) ([]point, error) {
		if len(centroids) != k {
			return nil, ErrCentroidCount
		}
		out := make([]point, k)
		copy(out, centroids)
		return out, nil
	}
}

func TestPartitionSeparableClusters(t *testing.T) {
	data := []point{{0, 0}, {2, 0}, {10, 10}, {12, 10}}
	km := New(sqdist, mean, fixed(point{0, 0}, point{10, 10}), WithWorkers(8))
	got, err := km.Partition(data, 2)
	if err != nil {
		t.Fatalf("Partition: %v", err)
	}
	wantCentroids := []point{{1, 0}, {11, 10}}
	if len(got.Centroids) != len(wantCentroids) {
		t.Fatalf("centroids = %v", got.Centroids)
	}
	for i := range wantCentroids {
		if got.Centroids[i] != wantCentroids[i] {
			t.Fatalf("centroid %d = %v, want %v", i, got.Centroids[i], wantCentroids[i])
		}
	}
	wantLabels := []int{0, 0, 1, 1}
	if len(got.Labels) != len(wantLabels) {
		t.Fatalf("labels = %v", got.Labels)
	}
	for i := range wantLabels {
		if got.Labels[i] != wantLabels[i] {
			t.Fatalf("labels = %v, want %v", got.Labels, wantLabels)
		}
	}
	if got.Iterations != 2 {
		t.Fatalf("iterations = %d, want 2", got.Iterations)
	}
	if got.Inertia != 4 {
		t.Fatalf("inertia = %v, want 4", got.Inertia)
	}
}

func TestPartitionMaxIterations(t *testing.T) {
	data := []point{{0, 0}, {2, 0}, {10, 10}, {12, 10}}
	km := New(sqdist, mean, fixed(point{0, 0}, point{10, 10}), WithMaxIterations(1))
	got, err := km.Partition(data, 2)
	if err != nil {
		t.Fatalf("Partition: %v", err)
	}
	if got.Iterations != 1 {
		t.Fatalf("iterations = %d, want 1", got.Iterations)
	}
}

func TestPartitionSingleCluster(t *testing.T) {
	data := []point{{2, 4}, {4, 6}}
	km := New(sqdist, mean, fixed(point{0, 0}))
	got, err := km.Partition(data, 1)
	if err != nil {
		t.Fatalf("Partition: %v", err)
	}
	if got.Centroids[0] != (point{3, 5}) {
		t.Fatalf("centroid = %v, want (3, 5)", got.Centroids[0])
	}
	for i, label := range got.Labels {
		if label != 0 {
			t.Fatalf("label %d = %d", i, label)
		}
	}
	if got.Inertia != 4 {
		t.Fatalf("inertia = %v, want 4", got.Inertia)
	}
}

func TestPartitionReseedsEmptyCluster(t *testing.T) {
	data := make([]point, 0, 11)
	for i := 0; i < 5; i++ {
		data = append(data, point{0, 1})
	}
	for i := 0; i < 5; i++ {
		data = append(data, point{10, 10})
	}
	data = append(data, point{50, 0})

	var averaged []point
	average := func(points []point) point {
		if len(points) == 0 {
			t.Fatal("averaged an empty cluster")
		}
		for _, p := range points {
			averaged = append(averaged, p)
			if p != (point{0, 1}) && p != (point{10, 10}) && p != (point{50, 0}) {
				t.Fatalf("averaged unexpected point %v", p)
			}
		}
		return mean(points)
	}

	km := New(sqdist, average, fixed(point{0, 1}, point{10, 10}, point{1000, 1000}))
	got, err := km.Partition(data, 3)
	if err != nil {
		t.Fatalf("Partition: %v", err)
	}
	if len(averaged) == 0 {
		t.Fatal("average was not called")
	}

	outlier := -1
	for i, c := range got.Centroids {
		if c == (point{50, 0}) {
			outlier = i
		}
		if c == (point{1000, 1000}) || c == (point{}) {
			t.Fatalf("centroid %d = %v", i, c)
		}
	}
	if outlier < 0 {
		t.Fatalf("outlier was not promoted to a centroid: %v", got.Centroids)
	}
	if got.Labels[len(data)-1] != outlier {
		t.Fatalf("outlier label = %d, want %d", got.Labels[len(data)-1], outlier)
	}
	if got.Inertia != 0 {
		t.Fatalf("inertia = %v, want 0", got.Inertia)
	}
	if got.Iterations != 3 {
		t.Fatalf("iterations = %d, want 3", got.Iterations)
	}
}

func TestPartitionParallelMatchesSequential(t *testing.T) {
	const n = 2000
	rng := rand.New(rand.NewPCG(42, 99))
	data := make([]point, n)
	for i := 0; i < n; i += 2 {
		data[i] = point{rng.Float64(), rng.Float64()}
		data[i+1] = point{10 + rng.Float64(), 10 + rng.Float64()}
	}
	init := fixed(data[0], data[1])

	seq, err := New(sqdist, mean, init, WithWorkers(1)).Partition(data, 2)
	if err != nil {
		t.Fatalf("sequential: %v", err)
	}
	par, err := New(sqdist, mean, init, WithWorkers(8)).Partition(data, 2)
	if err != nil {
		t.Fatalf("parallel: %v", err)
	}
	if seq.Iterations != par.Iterations || seq.Inertia != par.Inertia {
		t.Fatalf("seq = %+v, par = %+v", seq.Iterations, par.Iterations)
	}
	for i := range seq.Centroids {
		if seq.Centroids[i] != par.Centroids[i] {
			t.Fatalf("centroid %d seq %v par %v", i, seq.Centroids[i], par.Centroids[i])
		}
	}
	for i := range seq.Labels {
		if seq.Labels[i] != par.Labels[i] {
			t.Fatalf("label %d seq %d par %d", i, seq.Labels[i], par.Labels[i])
		}
	}
}

func TestPartitionErrors(t *testing.T) {
	data := []point{{1, 1}, {2, 2}}
	sentinel := errors.New("init failed")
	tests := []struct {
		name    string
		km      *KMeans[point]
		data    []point
		k       int
		wantErr error
	}{
		{
			name:    "nil metric",
			km:      New[point](nil, mean, fixed(point{})),
			data:    data,
			k:       1,
			wantErr: ErrNilMetric,
		},
		{
			name:    "nil average",
			km:      New(sqdist, nil, fixed(point{})),
			data:    data,
			k:       1,
			wantErr: ErrNilAverage,
		},
		{
			name:    "nil initializer",
			km:      New(sqdist, mean, nil),
			data:    data,
			k:       1,
			wantErr: ErrNilInitializer,
		},
		{
			name:    "empty",
			km:      New(sqdist, mean, fixed(point{})),
			data:    nil,
			k:       1,
			wantErr: ErrEmptyData,
		},
		{
			name:    "k zero",
			km:      New(sqdist, mean, fixed(point{})),
			data:    data,
			k:       0,
			wantErr: ErrInvalidK,
		},
		{
			name:    "k negative",
			km:      New(sqdist, mean, fixed(point{})),
			data:    data,
			k:       -1,
			wantErr: ErrInvalidK,
		},
		{
			name:    "k too large",
			km:      New(sqdist, mean, fixed(point{})),
			data:    data,
			k:       3,
			wantErr: ErrInvalidK,
		},
		{
			name:    "epsilon",
			km:      New(sqdist, mean, fixed(point{}), WithEpsilon(-1)),
			data:    data,
			k:       1,
			wantErr: ErrInvalidEpsilon,
		},
		{
			name:    "nan epsilon",
			km:      New(sqdist, mean, fixed(point{}), WithEpsilon(math.NaN())),
			data:    data,
			k:       1,
			wantErr: ErrInvalidEpsilon,
		},
		{
			name:    "iterations",
			km:      New(sqdist, mean, fixed(point{}), WithMaxIterations(0)),
			data:    data,
			k:       1,
			wantErr: ErrInvalidIterations,
		},
		{
			name:    "workers",
			km:      New(sqdist, mean, fixed(point{}), WithWorkers(-1)),
			data:    data,
			k:       1,
			wantErr: ErrInvalidWorkers,
		},
		{
			name: "initializer",
			km: New(sqdist, mean, func([]point, int) ([]point, error) {
				return nil, sentinel
			}),
			data:    data,
			k:       1,
			wantErr: sentinel,
		},
		{
			name: "centroid count",
			km: New(sqdist, mean, func([]point, int) ([]point, error) {
				return []point{{1, 1}}, nil
			}),
			data:    data,
			k:       2,
			wantErr: ErrCentroidCount,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.km.Partition(tt.data, tt.k)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("err = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func TestPartitionDoesNotCallInitializerOnBadK(t *testing.T) {
	km := New(sqdist, mean, func([]point, int) ([]point, error) {
		t.Fatal("initializer should not run")
		return nil, nil
	})
	if _, err := km.Partition(nil, 2); !errors.Is(err, ErrEmptyData) {
		t.Fatalf("err = %v", err)
	}
}

func TestForgyDeterministic(t *testing.T) {
	data := make([]point, 20)
	for i := range data {
		data[i] = point{X: float64(i), Y: float64(i * i)}
	}
	run := func() []point {
		rng := rand.New(rand.NewPCG(7, 11))
		got, err := Forgy[point](rng)(data, 5)
		if err != nil {
			t.Fatalf("Forgy: %v", err)
		}
		return got
	}
	first := run()
	second := run()
	if len(first) != 5 {
		t.Fatalf("len = %d", len(first))
	}
	if !firstEqual(first, second) {
		t.Fatalf("same seed diverged: %v vs %v", first, second)
	}
	seen := map[point]struct{}{}
	for _, c := range first {
		if _, ok := seen[c]; ok {
			t.Fatalf("duplicate centroid %v", c)
		}
		seen[c] = struct{}{}
		found := false
		for _, p := range data {
			if p == c {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("centroid %v is not a data point", c)
		}
	}
}

func TestForgyErrors(t *testing.T) {
	_, err := Forgy[point](nil)(nil, 1)
	if !errors.Is(err, ErrEmptyData) {
		t.Fatalf("err = %v", err)
	}
	_, err = Forgy[point](nil)([]point{{1, 1}}, 2)
	if !errors.Is(err, ErrInvalidK) {
		t.Fatalf("err = %v", err)
	}
}

func TestKMeansPPSeparatesModes(t *testing.T) {
	data := make([]point, 0, 20)
	for i := 0; i < 10; i++ {
		data = append(data, point{X: 0})
	}
	for i := 0; i < 10; i++ {
		data = append(data, point{X: 100})
	}
	rng := rand.New(rand.NewPCG(1, 2))
	got, err := KMeansPP(sqdist, rng)(data, 2)
	if err != nil {
		t.Fatalf("KMeansPP: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len = %d", len(got))
	}
	if got[0].X == got[1].X {
		t.Fatalf("centroids collapsed to one mode: %v", got)
	}
}

func TestKMeansPPDeterministic(t *testing.T) {
	data := make([]point, 30)
	for i := range data {
		data[i] = point{X: float64(i)}
	}
	once := func() []point {
		got, err := KMeansPP(sqdist, rand.New(rand.NewPCG(3, 5)))(data, 4)
		if err != nil {
			t.Fatalf("KMeansPP: %v", err)
		}
		return got
	}
	first := once()
	second := once()
	if !firstEqual(first, second) {
		t.Fatalf("same seed diverged: %v vs %v", first, second)
	}
	seen := map[point]bool{}
	for _, c := range first {
		if seen[c] {
			t.Fatalf("duplicate centroid %v in %v", c, first)
		}
		seen[c] = true
	}
}

func TestKMeansPPErrors(t *testing.T) {
	_, err := KMeansPP[point](nil, nil)([]point{{1, 1}}, 1)
	if !errors.Is(err, ErrNilMetric) {
		t.Fatalf("err = %v", err)
	}
	_, err = KMeansPP(sqdist, nil)([]point{{1, 1}}, 0)
	if !errors.Is(err, ErrInvalidK) {
		t.Fatalf("err = %v", err)
	}
}

func TestSameSeedSamePartition(t *testing.T) {
	data := make([]point, 40)
	rng := rand.New(rand.NewPCG(9, 8))
	for i := range data {
		data[i] = point{X: rng.NormFloat64(), Y: rng.NormFloat64()}
	}
	run := func() Result[point] {
		km := New(sqdist, mean, Forgy[point](rand.New(rand.NewPCG(4, 4))), WithWorkers(4))
		got, err := km.Partition(data, 3)
		if err != nil {
			t.Fatalf("Partition: %v", err)
		}
		return got
	}
	first := run()
	second := run()
	if first.Iterations != second.Iterations || first.Inertia != second.Inertia {
		t.Fatalf("runs diverged: %+v vs %+v", first.Iterations, second.Iterations)
	}
	if !firstEqual(first.Centroids, second.Centroids) {
		t.Fatalf("centroids diverged")
	}
	for i := range first.Labels {
		if first.Labels[i] != second.Labels[i] {
			t.Fatalf("labels diverged at %d", i)
		}
	}
}

func firstEqual(a, b []point) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
