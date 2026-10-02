// Package kmeans clusters points with Lloyd's algorithm.
//
// The point type is generic. The caller supplies a distance, a center function,
// and a way to choose the initial centroids. The center must minimize the
// distance: for Euclidean data, pass squared Euclidean distance and the
// arithmetic mean.
package kmeans

import "errors"

// Metric measures how far x is from y.
// Smaller values mean closer points. Lloyd's algorithm converges when Average
// minimizes this value. For Euclidean geometry, return the squared distance so
// the arithmetic mean is the correct center.
//
// Partition may call Metric concurrently.
type Metric[T any] func(x, y T) float64

// Average returns the center of a non-empty cluster.
// The slice is valid only for the duration of the call; do not retain it.
// For squared Euclidean distance, this is the arithmetic mean.
type Average[T any] func(points []T) T

// Initializer chooses k starting centroids from data.
// Forgy and KMeansPP implement common strategies. A *rand.Rand passed to them
// must not be used from more than one goroutine.
type Initializer[T any] func(data []T, k int) ([]T, error)

// Result is one run of Lloyd's algorithm.
type Result[T any] struct {
	// Centroids are the cluster centers. Labels refer to indexes in this slice.
	Centroids []T
	// Labels[i] is the cluster index of input point i.
	Labels []int
	// Iterations is how many assignment-and-update rounds ran.
	Iterations int
	// Inertia is the sum of Metric from each point to its assigned centroid.
	Inertia float64
}

// Option configures a KMeans value created by New.
type Option func(*settings)

type settings struct {
	epsilon float64
	maxIter int
	workers int
}

// WithEpsilon sets the convergence tolerance on centroid movement.
// A round converges when every centroid shifts by at most epsilon, measured
// with Metric. The default is 1e-4. epsilon must be greater than or equal to zero.
func WithEpsilon(epsilon float64) Option {
	return func(s *settings) {
		s.epsilon = epsilon
	}
}

// WithMaxIterations sets the maximum number of assignment-and-update rounds.
// The default is 500. n must be greater than zero.
func WithMaxIterations(n int) Option {
	return func(s *settings) {
		s.maxIter = n
	}
}

// WithWorkers sets how many goroutines assign points to centroids.
// Zero, the default, uses GOMAXPROCS. Partition still runs the assignment
// sequentially when the input is too small to benefit from extra goroutines.
// n must be greater than or equal to zero.
func WithWorkers(n int) Option {
	return func(s *settings) {
		s.workers = n
	}
}

// KMeans runs Lloyd's algorithm for one point type.
// A value is safe for concurrent Partition calls when Metric, Average, and
// Initializer are themselves safe for concurrent use. Forgy and KMeansPP are
// not, if they share one *rand.Rand.
type KMeans[T any] struct {
	metric  Metric[T]
	average Average[T]
	init    Initializer[T]
	epsilon float64
	maxIter int
	workers int
}

// New returns a clusterer for point type T.
// metric, average, and init must be non-nil. Invalid options are reported by
// Partition, not by New.
func New[T any](metric Metric[T], average Average[T], init Initializer[T], opts ...Option) *KMeans[T] {
	s := settings{
		epsilon: 1e-4,
		maxIter: 500,
	}
	for _, opt := range opts {
		opt(&s)
	}
	return &KMeans[T]{
		metric:  metric,
		average: average,
		init:    init,
		epsilon: s.epsilon,
		maxIter: s.maxIter,
		workers: s.workers,
	}
}

var (
	// ErrEmptyData is returned when there are no points to cluster.
	ErrEmptyData = errors.New("kmeans: empty data")
	// ErrInvalidK is returned when k is less than 1 or greater than the number of points.
	ErrInvalidK = errors.New("kmeans: invalid k")
	// ErrNilMetric is returned when the distance function is nil.
	ErrNilMetric = errors.New("kmeans: nil metric")
	// ErrNilAverage is returned when the center function is nil.
	ErrNilAverage = errors.New("kmeans: nil average")
	// ErrNilInitializer is returned when the centroid initializer is nil.
	ErrNilInitializer = errors.New("kmeans: nil initializer")
	// ErrInvalidEpsilon is returned when epsilon is negative or NaN.
	ErrInvalidEpsilon = errors.New("kmeans: invalid epsilon")
	// ErrInvalidIterations is returned when the iteration limit is less than 1.
	ErrInvalidIterations = errors.New("kmeans: invalid max iterations")
	// ErrInvalidWorkers is returned when the worker count is negative.
	ErrInvalidWorkers = errors.New("kmeans: invalid workers")
	// ErrCentroidCount is returned when an initializer yields a number of centroids other than k.
	ErrCentroidCount = errors.New("kmeans: initializer returned the wrong number of centroids")
)

func validateSize(n, k int) error {
	if n == 0 {
		return ErrEmptyData
	}
	if k < 1 || k > n {
		return ErrInvalidK
	}
	return nil
}
