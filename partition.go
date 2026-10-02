package kmeans

import (
	"math"
	"runtime"
	"sync"
)

// minChunk is the smallest assignment batch that is worth a separate goroutine.
const minChunk = 64

// Partition clusters data into k groups.
// k must be between 1 and len(data). Equal distances assign the point to the
// centroid with the smaller index.
//
// An empty cluster is not averaged: its centroid is moved to the point farthest
// from its own centroid, preferring a point that is not alone in its cluster.
func (km *KMeans[T]) Partition(data []T, k int) (Result[T], error) {
	var zero Result[T]
	if km.metric == nil {
		return zero, ErrNilMetric
	}
	if km.average == nil {
		return zero, ErrNilAverage
	}
	if km.init == nil {
		return zero, ErrNilInitializer
	}
	if km.epsilon < 0 || math.IsNaN(km.epsilon) {
		return zero, ErrInvalidEpsilon
	}
	if km.maxIter < 1 {
		return zero, ErrInvalidIterations
	}
	if km.workers < 0 {
		return zero, ErrInvalidWorkers
	}
	if err := validateSize(len(data), k); err != nil {
		return zero, err
	}

	raw, err := km.init(data, k)
	if err != nil {
		return zero, err
	}
	if len(raw) != k {
		return zero, ErrCentroidCount
	}
	centroids := make([]T, k)
	copy(centroids, raw)

	labels := make([]int, len(data))
	iterations := 0
	for {
		km.assign(data, centroids, labels)
		next, maxShift, reseeded := km.update(data, centroids, labels, k)
		centroids = next
		iterations++
		if (!reseeded && maxShift <= km.epsilon) || iterations >= km.maxIter {
			break
		}
	}

	km.assign(data, centroids, labels)
	return Result[T]{
		Centroids:  centroids,
		Labels:     labels,
		Iterations: iterations,
		Inertia:    km.inertia(data, centroids, labels),
	}, nil
}

func (km *KMeans[T]) assign(data, centroids []T, labels []int) {
	n := len(data)
	workers := km.workerCount(n)
	if workers == 1 {
		km.assignRange(data, centroids, labels, 0, n)
		return
	}

	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > n {
			end = n
		}
		go func(start, end int) {
			defer wg.Done()
			if start < end {
				km.assignRange(data, centroids, labels, start, end)
			}
		}(start, end)
	}
	wg.Wait()
}

func (km *KMeans[T]) assignRange(data, centroids []T, labels []int, start, end int) {
	k := len(centroids)
	for i := start; i < end; i++ {
		best := 0
		bestD := km.metric(data[i], centroids[0])
		for j := 1; j < k; j++ {
			d := km.metric(data[i], centroids[j])
			if d < bestD {
				bestD = d
				best = j
			}
		}
		labels[i] = best
	}
}

// update rebuilds centroids from the current labels.
// reseeded is true when at least one cluster was empty.
func (km *KMeans[T]) update(data, centroids []T, labels []int, k int) (next []T, maxShift float64, reseeded bool) {
	counts := make([]int, k)
	for _, label := range labels {
		counts[label]++
	}

	groups := make([][]T, k)
	for c := range groups {
		if counts[c] > 0 {
			groups[c] = make([]T, 0, counts[c])
		}
	}
	for i, label := range labels {
		groups[label] = append(groups[label], data[i])
	}

	next = make([]T, k)
	empty := make([]int, 0)
	maxShift = math.Inf(-1)
	for c := 0; c < k; c++ {
		if counts[c] == 0 {
			empty = append(empty, c)
			continue
		}
		next[c] = km.average(groups[c])
		shift := km.metric(next[c], centroids[c])
		if math.IsNaN(shift) || shift > maxShift {
			maxShift = shift
		}
	}
	if len(empty) == 0 {
		return next, maxShift, false
	}
	km.reseed(data, centroids, labels, counts, next, empty)
	return next, maxShift, true
}

// reseed moves each empty centroid onto a data point, farthest first from the
// centroid it is currently assigned to. Points that are the only member of a
// cluster are kept when some other cluster can spare a point.
func (km *KMeans[T]) reseed(data, old []T, labels, counts []int, next []T, empty []int) {
	used := make(map[int]struct{}, len(empty))
	multi := false
	for _, count := range counts {
		if count > 1 {
			multi = true
			break
		}
	}
	for _, c := range empty {
		best := -1
		bestD := math.Inf(-1)
		for i := range data {
			if _, taken := used[i]; taken {
				continue
			}
			if multi && counts[labels[i]] < 2 {
				continue
			}
			d := km.metric(data[i], old[labels[i]])
			if d > bestD {
				bestD = d
				best = i
			}
		}
		if best < 0 {
			for i := range data {
				if _, taken := used[i]; !taken {
					best = i
					break
				}
			}
		}
		if best < 0 {
			next[c] = old[c]
			continue
		}
		next[c] = data[best]
		used[best] = struct{}{}
	}
}

func (km *KMeans[T]) inertia(data, centroids []T, labels []int) float64 {
	var sum float64
	for i := range data {
		sum += km.metric(data[i], centroids[labels[i]])
	}
	return sum
}

func (km *KMeans[T]) workerCount(n int) int {
	workers := km.workers
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers > n {
		workers = n
	}
	if workers > 1 {
		limit := n / minChunk
		if limit < 1 {
			limit = 1
		}
		if workers > limit {
			workers = limit
		}
	}
	return workers
}
