package kmeans

import "math/rand/v2"

// Forgy returns an initializer that picks k distinct points from data.
// A nil rng uses the global random source.
func Forgy[T any](rng *rand.Rand) Initializer[T] {
	return func(data []T, k int) ([]T, error) {
		if err := validateSize(len(data), k); err != nil {
			return nil, err
		}
		idx := sampleDistinct(len(data), k, rng)
		centroids := make([]T, k)
		for i, j := range idx {
			centroids[i] = data[j]
		}
		return centroids, nil
	}
}

// KMeansPP returns an initializer that spreads centroids with the k-means++ rule.
// Each new centroid is chosen with probability proportional to Metric against
// the nearest centroid already chosen. Pass squared Euclidean distance so that
// those weights are squared distances. A nil rng uses the global random source.
func KMeansPP[T any](metric Metric[T], rng *rand.Rand) Initializer[T] {
	return func(data []T, k int) ([]T, error) {
		if metric == nil {
			return nil, ErrNilMetric
		}
		n := len(data)
		if err := validateSize(n, k); err != nil {
			return nil, err
		}

		chosen := make([]bool, n)
		dist := make([]float64, n)
		centroids := make([]T, 0, k)

		first := intn(rng, n)
		chosen[first] = true
		centroids = append(centroids, data[first])

		for len(centroids) < k {
			last := centroids[len(centroids)-1]
			var sum float64
			for i := range data {
				if chosen[i] {
					dist[i] = 0
					continue
				}
				d := metric(data[i], last)
				if d < 0 {
					d = 0
				}
				if len(centroids) == 1 || d < dist[i] {
					dist[i] = d
				}
				sum += dist[i]
			}

			idx := pickWeighted(dist, sum, rng)
			if idx < 0 || chosen[idx] {
				idx = firstFree(chosen)
			}
			if idx < 0 {
				return nil, ErrInvalidK
			}
			chosen[idx] = true
			centroids = append(centroids, data[idx])
		}
		return centroids, nil
	}
}

func sampleDistinct(n, k int, rng *rand.Rand) []int {
	swap := make(map[int]int, k*2)
	out := make([]int, k)
	for i := 0; i < k; i++ {
		j := i + intn(rng, n-i)
		vi, ok := swap[i]
		if !ok {
			vi = i
		}
		vj, ok := swap[j]
		if !ok {
			vj = j
		}
		swap[i] = vj
		swap[j] = vi
		out[i] = vj
	}
	return out
}

func pickWeighted(dist []float64, sum float64, rng *rand.Rand) int {
	if sum <= 0 {
		return -1
	}
	target := floatn(rng) * sum
	var acc float64
	last := -1
	for i, d := range dist {
		if d <= 0 {
			continue
		}
		last = i
		acc += d
		if acc >= target {
			return i
		}
	}
	return last
}

func firstFree(chosen []bool) int {
	for i, taken := range chosen {
		if !taken {
			return i
		}
	}
	return -1
}

func intn(rng *rand.Rand, n int) int {
	if rng == nil {
		return rand.IntN(n)
	}
	return rng.IntN(n)
}

func floatn(rng *rand.Rand) float64 {
	if rng == nil {
		return rand.Float64()
	}
	return rng.Float64()
}
