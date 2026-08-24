#include "NectoPolicy.h"

#include <RLGymCPP/CommonValues.h>

#include <torch/script.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

using namespace RLGC;

namespace Dash {

namespace {

// necto_obs.py `_norm`. Note positions are divided by 2300 as well as
// velocities -- that is Necto's convention, not a transcription slip.
constexpr float NORM[NectoPolicy::FEATURES] = {
	1, 1, 1, 1, 1,          // 0-4   type flags
	2300, 2300, 2300,       // 5-7   position
	2300, 2300, 2300,       // 8-10  linear velocity
	1, 1, 1,                // 11-13 forward
	1, 1, 1,                // 14-16 up
	5.5f, 5.5f, 5.5f,       // 17-19 angular velocity
	1, 1, 1, 1              // 20-23 boost, timer, on ground, has flip
};

// necto_obs.py `_invert`: negate x and y of every vector triple, keep z.
constexpr float INVERT[NectoPolicy::FEATURES] = {
	1,  1,  1, 1, 1, //
	-1, -1, 1,       // position
	-1, -1, 1,       // linear velocity
	-1, -1, 1,       // forward
	-1, -1, 1,       // up
	-1, -1, 1,       // angular velocity
	1,  1,  1, 1     //
};

constexpr int NUM_HEADS = 5;
constexpr int MAX_HEAD_SIZE = 3;

} // namespace

struct NectoPolicy::Module {
	torch::jit::script::Module mod;
};

NectoPolicy::NectoPolicy(const std::filesystem::path &modelPath, float beta,
						 int64_t seed)
	: beta(std::clamp(beta, 0.f, 1.f)),
	  rng(seed < 0 ? std::random_device{}() : static_cast<uint32_t>(seed)) {
	if (!std::filesystem::is_regular_file(modelPath))
		throw std::runtime_error("NectoPolicy: no model file at " +
								 modelPath.string());

	// Necto runs on CPU on purpose: training owns the GPU, and a batch of ~50
	// costs ~3ms here, which is nothing against the sim and the PPO update.
	//
	// Note it does NOT call at::set_num_threads(). Necto's own agent.py does,
	// because under RLBot it must not steal the game's CPU -- but that setting
	// is process-global in libtorch, so calling it here throttled PPO's own
	// inference and update to a single thread. Measured cost of getting this
	// wrong: collection throughput roughly halved.

	module = std::make_unique<Module>();
	try {
		module->mod = torch::jit::load(modelPath.string(), torch::kCPU);
		module->mod.eval();
	} catch (const std::exception &e) {
		throw std::runtime_error("NectoPolicy: failed to load " +
								 modelPath.string() + ": " + e.what());
	}

	// agent.py: logits *= log((beta + 1) / (1 - beta), 3). Falls out to exactly
	// 1.0 at beta = 0.5, which is why that value is plain on-policy sampling.
	logitScale = (this->beta >= 1.f || this->beta <= 0.f)
					 ? 0.f
					 : std::log((this->beta + 1.f) / (1.f - this->beta)) /
						   std::log(3.f);
}

NectoPolicy::~NectoPolicy() = default;

int NectoPolicy::TokenCount(const GameState &state) {
	return 1 + static_cast<int>(state.players.size()) +
		   CommonValues::BOOST_LOCATIONS_AMOUNT;
}

void NectoPolicy::BuildObs(const GameState &gs, int playerIdx,
						   const Action &prevAction, float *qOut,
						   float *kvOut) {
	const int numPlayers = static_cast<int>(gs.players.size());
	const int n = TokenCount(gs);

	if (playerIdx < 0 || playerIdx >= numPlayers)
		throw std::runtime_error("NectoPolicy::BuildObs: player index out of range");
	if (static_cast<int>(gs.boostPadTimers.size()) <
		CommonValues::BOOST_LOCATIONS_AMOUNT)
		throw std::runtime_error("NectoPolicy::BuildObs: game state has no boost pad data");

	std::fill(kvOut, kvOut + static_cast<size_t>(n) * FEATURES, 0.f);

	auto row = [&](int i) { return kvOut + static_cast<size_t>(i) * FEATURES; };
	auto setVec = [](float *r, int col, const Vec &v) {
		r[col] = v.x;
		r[col + 1] = v.y;
		r[col + 2] = v.z;
	};

	{ // Ball. Leaves the forward/up columns zero, as Necto does.
		float *r = row(0);
		r[3] = 1.f; // is_ball
		setVec(r, 5, gs.ball.pos);
		setVec(r, 8, gs.ball.vel);
		setVec(r, 17, gs.ball.angVel);
	}

	// Cars. Team flags are ABSOLUTE here (blue -> col 1, orange -> col 2); the
	// swap for an orange main car happens below, in the Python's order. Do not
	// shortcut this by computing the flags relative to the main car -- the
	// inversion below would then apply on top of an already-relative encoding.
	for (int i = 0; i < numPlayers; i++) {
		const Player &p = gs.players[i];
		float *r = row(1 + i);
		r[p.team == Team::BLUE ? 1 : 2] = 1.f;
		setVec(r, 5, p.pos);
		setVec(r, 8, p.vel);
		setVec(r, 11, p.rotMat.forward);
		setVec(r, 14, p.rotMat.up);
		setVec(r, 17, p.angVel);
		r[20] = p.boost / 100.f; // CarState::boost is 0-100, Necto wants 0-1
		// The sim's real respawn timer. necto_obs.py re-seeds this to 3 whenever
		// it reaches 0, which is an artifact of its RLBot port -- the model was
		// trained on the real thing.
		r[21] = p.demoRespawnTimer / 10.f;
		r[22] = p.isOnGround ? 1.f : 0.f;
		r[23] = p.HasFlipOrJump() ? 1.f : 0.f;
	}

	// Boost pads. Ordering does not matter -- the model attends over the token
	// set and each pad carries its own position -- so long as position, size and
	// timer for a row all come from the same pad.
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		const Vec &loc = CommonValues::BOOST_LOCATIONS[i];
		float *r = row(1 + numPlayers + i);
		r[4] = 1.f; // is_boost
		setVec(r, 5, loc);
		r[20] = 0.12f + 0.88f * (loc.z > 72.f ? 1.f : 0.f); // big pads sit at z=73
		// Index directly: GameState::GetBoostPadTimers() has its `inverted` flag
		// swapped upstream.
		r[21] = gs.boostPadTimers[i] / 10.f;
	}

	for (int i = 0; i < n; i++) {
		float *r = row(i);
		for (int c = 0; c < FEATURES; c++)
			r[c] /= NORM[c];
	}

	const int mainRow = 1 + playerIdx;
	row(mainRow)[0] = 1.f; // is_main

	// Orange plays the mirrored field.
	if (gs.players[playerIdx].team == Team::ORANGE) {
		for (int i = 0; i < n; i++) {
			float *r = row(i);
			std::swap(r[1], r[2]);
			for (int c = 0; c < FEATURES; c++)
				r[c] *= INVERT[c];
		}
	}

	// q keeps ABSOLUTE position and velocity, and must be taken before kv is
	// made relative below.
	std::copy(row(mainRow), row(mainRow) + FEATURES, qOut);
	for (int k = 0; k < static_cast<int>(Action::ELEM_AMOUNT); k++)
		qOut[FEATURES + k] = prevAction[k];

	// kv positions and velocities become relative to the main car.
	for (int i = 0; i < n; i++) {
		float *r = row(i);
		for (int c = 5; c < 11; c++)
			r[c] -= qOut[c];
	}
}

Action NectoPolicy::SampleAction(const float *const *heads,
								 const int *headSizes) {
	int picked[NUM_HEADS] = {};

	for (int h = 0; h < NUM_HEADS; h++) {
		const int k = headSizes[h];
		const float *logits = heads[h];

		if (beta >= 1.f) { // argmax
			int best = 0;
			for (int c = 1; c < k; c++)
				if (logits[c] > logits[best])
					best = c;
			picked[h] = best;
			continue;
		}

		if (beta <= 0.f) { // uniform
			picked[h] = std::uniform_int_distribution<int>(0, k - 1)(rng);
			continue;
		}

		float scaled[MAX_HEAD_SIZE];
		float maxLogit = -std::numeric_limits<float>::infinity();
		for (int c = 0; c < k; c++) {
			scaled[c] = logits[c] * logitScale;
			maxLogit = std::max(maxLogit, scaled[c]);
		}

		float sum = 0.f;
		for (int c = 0; c < k; c++) {
			scaled[c] = std::exp(scaled[c] - maxLogit);
			sum += scaled[c];
		}

		const float u =
			std::uniform_real_distribution<float>(0.f, 1.f)(rng) * sum;
		float acc = 0.f;
		picked[h] = k - 1;
		for (int c = 0; c < k; c++) {
			acc += scaled[c];
			if (u <= acc) {
				picked[h] = c;
				break;
			}
		}
	}

	// agent.py's mapping. Heads 0 and 1 are ternary and shift to {-1, 0, 1};
	// the rest are binary. Note pitch reuses head 0 and yaw/roll split head 1
	// on the handbrake bit -- that is why Necto's action space does not line up
	// with RLGymCPP's 90-entry lookup table.
	const float throttle = static_cast<float>(picked[0] - 1);
	const float steer = static_cast<float>(picked[1] - 1);
	const float jump = static_cast<float>(picked[2]);
	const float boost = static_cast<float>(picked[3]);
	const float handbrake = static_cast<float>(picked[4]);

	Action act = {};
	act.throttle = throttle;
	act.steer = steer;
	act.pitch = throttle;
	act.yaw = steer * (1.f - handbrake);
	act.roll = steer * handbrake;
	act.jump = jump;
	act.boost = boost;
	act.handbrake = handbrake;
	return act;
}

void NectoPolicy::InferBatch(const std::vector<NectoRequest> &requests,
							 std::vector<Action> &outActions) {
	outActions.assign(requests.size(), Action{});
	if (requests.empty())
		return;

	// Group by token count so every batch is exact. Padding ragged batches is
	// close but not exact -- the traced model's mask is not fully wired, so
	// padded rows still perturb the logits by ~0.03. At 1v1 this is one group.
	std::map<int, std::vector<int>> byTokens;
	for (int i = 0; i < static_cast<int>(requests.size()); i++)
		byTokens[TokenCount(*requests[i].state)].push_back(i);

	for (const auto &[n, idxs] : byTokens) {
		const int b = static_cast<int>(idxs.size());

		qBuf.assign(static_cast<size_t>(b) * Q_SIZE, 0.f);
		kvBuf.assign(static_cast<size_t>(b) * n * FEATURES, 0.f);
		maskBuf.assign(static_cast<size_t>(b) * n, 0.f);

		for (int j = 0; j < b; j++) {
			const NectoRequest &req = requests[idxs[j]];
			BuildObs(*req.state, req.playerIdx, req.prevAction,
					 qBuf.data() + static_cast<size_t>(j) * Q_SIZE,
					 kvBuf.data() + static_cast<size_t>(j) * n * FEATURES);
		}

		c10::IValue out;
		{
			torch::NoGradGuard noGrad;
			std::vector<c10::IValue> tup{
				torch::from_blob(qBuf.data(), {b, 1, Q_SIZE}, torch::kFloat32),
				torch::from_blob(kvBuf.data(), {b, n, FEATURES},
								 torch::kFloat32),
				torch::from_blob(maskBuf.data(), {b, n}, torch::kFloat32)};
			out = module->mod.forward({c10::ivalue::Tuple::create(tup)});
		}

		// The model returns (logit heads, attention weights). The heads come
		// back as a tuple of five tensors shaped [batch, 1, k].
		const auto outer = out.toTuple()->elements();
		if (outer.empty())
			throw std::runtime_error("NectoPolicy: model returned an empty tuple");

		std::vector<torch::Tensor> heads;
		if (outer[0].isTuple()) {
			for (const auto &e : outer[0].toTuple()->elements())
				heads.push_back(e.toTensor().contiguous());
		} else if (outer[0].isTensorList()) {
			for (const auto &t : outer[0].toTensorList())
				heads.push_back(torch::Tensor(t).contiguous());
		} else {
			throw std::runtime_error("NectoPolicy: unexpected head container '" +
									 std::string(outer[0].tagKind()) + "'");
		}

		if (static_cast<int>(heads.size()) != NUM_HEADS)
			throw std::runtime_error("NectoPolicy: expected " +
									 std::to_string(NUM_HEADS) + " heads, got " +
									 std::to_string(heads.size()));

		const float *headPtrs[NUM_HEADS];
		int headSizes[NUM_HEADS];
		for (int h = 0; h < NUM_HEADS; h++) {
			headSizes[h] = static_cast<int>(heads[h].size(-1));
			if (headSizes[h] > MAX_HEAD_SIZE)
				throw std::runtime_error("NectoPolicy: head wider than expected");
		}

		for (int j = 0; j < b; j++) {
			for (int h = 0; h < NUM_HEADS; h++)
				headPtrs[h] = heads[h].data_ptr<float>() +
							  static_cast<size_t>(j) * headSizes[h];
			outActions[idxs[j]] = SampleAction(headPtrs, headSizes);
		}
	}
}

} // namespace Dash
