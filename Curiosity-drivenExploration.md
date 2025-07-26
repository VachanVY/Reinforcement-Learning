# Curiosity-driven Exploration by Self-supervised Prediction
* <img width="550" height="858" alt="image" src="https://github.com/user-attachments/assets/8da0daa0-e27d-403c-8561-86d6a84114a8" />
* <img width="652" height="509" alt="image" src="https://github.com/user-attachments/assets/103e997f-9772-4d7d-8ca0-f2f644eadec2" />
* More generally, curiosity is a way of learning new skills which might come handy for pursuing rewards in the future
* Measuring “novelty” requires a statistical model of the dis-
tribution of the environmental states, whereas measuring
prediction error/uncertainty requires building a model of
environmental dynamics that predicts the next state (st+1 )
given the current state (st ) and the action (at ) executed
at time t.
* This work belongs to the broad category of methods that
generate an intrinsic reward signal based on how hard it is
for the agent to predict the consequences of its own actions,
i.e. predict the next state given the current state and the ex
ecuted action
* That is, instead of
making predictions in the raw sensory space (e.g. pixels),
we transform the sensory input into a feature space where
only the information relevant to the action performed by
the agent is represented. We learn this feature space using
self-supervision – training a neural network on a proxy in-
verse dynamics task of predicting the agent’s action given
its current and next states.
* We then use this feature space to train a forward dynamics
model that predicts the feature representation of the next
state, given the feature representation of the current state
and the action. We provide the prediction error of the for-
ward dynamics model to the agent as an intrinsic reward to
encourage its curiosity.
* In our opinion, cu-
riosity has two other fundamental uses. Curiosity helps an
agent explore its environment in the quest for new knowl-
edge (a desirable characteristic of exploratory behavior is
that it should improve as the agent gains more knowledge).
Further, curiosity is a mechanism for an agent to learn skills
that might be helpful in future scenarios.
* <img width="647" height="770" alt="image" src="https://github.com/user-attachments/assets/2784beea-69fa-4bdb-a9be-2a532892d1b2" />
* <img width="644" height="390" alt="image" src="https://github.com/user-attachments/assets/9199d789-3ad4-4c61-84e1-aeb7042fb36c" />
* <img width="1316" height="718" alt="image" src="https://github.com/user-attachments/assets/20554145-fa0a-4cda-ade6-4b2cd091b7fd" />
* <img width="652" height="640" alt="image" src="https://github.com/user-attachments/assets/cf8351e6-7435-4060-929c-0d3e00a865ec" />
* <img width="804" height="1072" alt="image" src="https://github.com/user-attachments/assets/cc33ef43-f6e6-494f-a820-6c488227360e" />
* <img width="733" height="634" alt="image" src="https://github.com/user-attachments/assets/c84d75fe-8670-4ec1-8445-904a3945daa1" />
* <img width="1492" height="667" alt="image" src="https://github.com/user-attachments/assets/e629a61f-84d2-459e-8cfb-8095c2f6cc51" />
* 
