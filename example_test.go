package kmeans_test

import (
	"fmt"

	"github.com/dennistrukhin/go-kmeans"
)

func ExampleKMeans_Partition() {
	data := [][2]float64{{0, 0}, {2, 0}, {10, 10}, {12, 10}}
	km := kmeans.New(
		func(a, b [2]float64) float64 {
			dx, dy := a[0]-b[0], a[1]-b[1]
			return dx*dx + dy*dy
		},
		func(points [][2]float64) [2]float64 {
			var sum [2]float64
			for _, p := range points {
				sum[0] += p[0]
				sum[1] += p[1]
			}
			n := float64(len(points))
			return [2]float64{sum[0] / n, sum[1] / n}
		},
		func([][2]float64, int) ([][2]float64, error) {
			return [][2]float64{{0, 0}, {10, 10}}, nil
		},
	)
	result, err := km.Partition(data, 2)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%.0f,%.0f\n", result.Centroids[0][0], result.Centroids[0][1])
	fmt.Printf("%.0f,%.0f\n", result.Centroids[1][0], result.Centroids[1][1])
	fmt.Println(result.Labels)
	// Output:
	// 1,0
	// 11,10
	// [0 0 1 1]
}
