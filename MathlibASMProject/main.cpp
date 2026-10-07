#include <iostream>
#include <vector>
#include "../MathLibrary/Nanobench/nanobench.h"
#include "../MathLibrary/Vector2SMID.h"   // adapte le chemin vers ton header

int main() {
    using ankerl::nanobench::doNotOptimizeAway;
    using math::Vector2;

    // 1. Préparation des données (hors mesure)
    ankerl::nanobench::Rng rng(42);   // seed fixe : mesures reproductibles
    const size_t N = 1024;            // puissance de 2 pour utiliser un masque
    std::vector<Vector2> a(N), b(N);

    for (size_t j = 0; j < N; ++j) {
        a[j] = Vector2(float(rng.uniform01()) * 10.0f, float(rng.uniform01()) * 10.0f);
        b[j] = Vector2(float(rng.uniform01()) * 10.0f, float(rng.uniform01()) * 10.0f);
    }

    // 2. Un seul Bench pour avoir un seul tableau
    ankerl::nanobench::Bench bench;
    bench.title("Vector2").performanceCounters(false).warmup(100);

    size_t i = 0;   // index partagé, il tourne de 0 à N-1 grâce au masque

    // --- Opérateurs ---
    bench.run("operator+", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(a[k] + b[k]);
        });

    bench.run("operator-", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(a[k] - b[k]);
        });

    bench.run("operator* (scalaire)", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(a[k] * 2.5f);
        });

    // --- Produit scalaire et normes ---
    bench.run("dot", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(a[k].dot(b[k]));
        });

    bench.run("sqrMagnitude", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(a[k].sqrMagnitude());
        });

    bench.run("magnitude", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(a[k].magnitude());
        });

    bench.run("normalized", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(a[k].normalized());
        });

    // --- Fonctions statiques ---
    bench.run("distance", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(Vector2::distance(a[k], b[k]));
        });

    bench.run("Lerp", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(Vector2::Lerp(a[k], b[k], 0.3f));
        });

    bench.run("angle", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(Vector2::angle(a[k], b[k]));
        });

    bench.run("MoveTowards", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(Vector2::MoveTowards(a[k], b[k], 1.0f));
        });

    // --- Utilitaires ---
    bench.run("Reflect", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(a[k].Reflect(b[k]));
        });

    bench.run("Perpendicular", [&] {
        size_t k = i++ & (N - 1);
        doNotOptimizeAway(a[k].Perpendicular());
        });
}