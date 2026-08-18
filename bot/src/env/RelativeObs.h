#pragma once

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

namespace Hive {

// DefaultObsPadded plus car-frame relative geometry for the ball and every
// other car.
//
// WHY. The p8ref observation was entirely absolute, world-frame. To compute
// `v . dirToBall` -- the quantity 55% of the reward mass pays for -- the
// network had to subtract two absolute 3-vectors, normalise, and dot with a
// third; and to ACT on the answer it had to further rotate the result into its
// own frame using the forward and up vectors. That is four separate 3-vectors
// and a change of basis, learned from scratch, before a single useful decision
// can be made.
//
// Ground driving is nearly planar and tolerates that. Aerial control is far
// more demanding of it, and p8ref's bot resolved the tension by playing an
// almost purely 2D game: `Player/Touch Height` 149, `Touch/Above 450` 0.035,
// high balls reached by driving up the wall rather than by leaving the ground.
// Handing the network the relative geometry directly is the cheapest available
// test of whether representation cost is what shapes that preference.
//
// This is not novel. rlgym-tools ships `relative_physics()`
// (`(target.position - origin.position) @ rot`) for exactly this, and Necto,
// Nexto and current community observations all carry relative features.
// Zealan's guide: "I've found you can get moderately better results if you add
// car-relative positions and velocities".
//
// STATELESS, deliberately. Action stacking (the other half of the modern obs)
// needs per-car history and therefore an episode reset, and `ObsBuilder::Reset`
// is called by EnvSet during training but by NOTHING on the RLBot deployment
// path. A stateful builder would train and deploy differently with no symptom
// -- the bot would load, play, and be quietly worse. That is its own change,
// with its own reset plumbing and its own run.
class RelativeObs : public RLGC::ObsBuilder {
public:
	// Car-frame positions are not field-aligned, so a single scalar is used for
	// all three components rather than DefaultObs's per-axis coefficients.
	// BACK_WALL_Y puts a cross-field separation near 1 and the arena diagonal
	// near 2.6.
	static constexpr float RELATIVE_POS_SCALE = RLGC::CommonValues::BACK_WALL_Y;

	explicit RelativeObs(int maxPlayers) : maxPlayers(maxPlayers) {}

	RLGC::FList BuildObs(const RLGC::Player& player, const RLGC::GameState& state) override;

	// Number of floats one relative block contributes. Exposed for tests.
	static constexpr int RELATIVE_BLOCK = 10;

private:
	int maxPlayers;
};

} // namespace Hive
