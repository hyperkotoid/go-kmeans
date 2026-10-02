# K-Means for Go

[![Go Report Card](https://goreportcard.com/badge/github.com/hyperkotoid/go-kmeans)](https://goreportcard.com/report/github.com/hyperkotoid/go-kmeans)

Generic Lloyd's algorithm. The point type is arbitrary: the library needs a distance, a cluster center, and the initial centroids.

One run returns:

- centroids
- a cluster label for each input point
- the iteration count
- inertia: the sum of distances from points to their assigned centroids

```bash
go get github.com/hyperkotoid/go-kmeans
```

### Example

```go
package main

import (
	"fmt"

	"github.com/hyperkotoid/go-kmeans"
)

type Point struct{ X, Y float64 }

func main() {
	data := []Point{{0, 0}, {2, 0}, {10, 10}, {12, 10}}
	km := kmeans.New(
		func(a, b Point) float64 {
			dx, dy := a.X-b.X, a.Y-b.Y
			return dx*dx + dy*dy
		},
		func(points []Point) Point {
			var sum Point
			for _, p := range points {
				sum.X += p.X
				sum.Y += p.Y
			}
			n := float64(len(points))
			return Point{sum.X / n, sum.Y / n}
		},
		func([]Point, int) ([]Point, error) {
			return []Point{{0, 0}, {10, 10}}, nil
		},
	)
	result, err := km.Partition(data, 2)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Centroids)
	fmt.Println(result.Labels)
}
```

`Forgy` (k distinct data points) and `KMeansPP` provide random initialization. Both take a `*rand.Rand`. Do not use one generator from several goroutines.

### Contract

`Metric` is the distance between two points. For Euclidean data, pass squared distance, without `Sqrt`: the arithmetic mean is then the correct center, and inertia is the sum of squared errors. `Partition` may call `Metric` concurrently.

`Average` is the center of a non-empty cluster. It must minimize `Metric`, or the algorithm is not guaranteed to converge. The point slice is valid only for the duration of the call.

`Initializer` chooses the k starting centroids. `WithEpsilon` (default `1e-4`) sets the centroid-shift tolerance, `WithMaxIterations` (default 500) sets the iteration limit, and `WithWorkers` sets how many goroutines assign points. Short inputs are assigned on a single thread.

An empty cluster is not averaged. Its centroid moves to the point farthest from that point's current centroid, preferring a point from a cluster that has at least two points.

`Partition` returns an error when there is no data, `k` is outside `1..len(data)`, a required function is missing, or an option is invalid.
