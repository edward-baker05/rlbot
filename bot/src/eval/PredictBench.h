#pragma once

namespace Dash {

// Times the CPU env-stepping path with and without the prediction block.
//
// Deliberately not a training run: t1 occupies the GPU, and the cost this gate
// is about is CPU ball simulation, which a policy-free loop measures cleanly.
int RunPredictBench(int numArenas, int steps);

}  // namespace Dash
