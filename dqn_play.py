import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from typing import Callable
from itertools import count

from dqn import get_model, prepro

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=int, default=0)
    args = parser.parse_args()

    ENV_NAME = "ALE/Pong-v5"
    MODEL_WEI_PATH = "ckpt/model_final21551787.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(ENV_NAME, render_mode="rgb_array", difficulty=args.difficulty)

    def plot_environment(env:gym.Env, figsize:tuple[int, int]=(5, 4)):
        plt.figure(figsize=figsize)
        img = env.render()
        plt.imshow(img)
        plt.axis("off")
        return img

    def update_scene(num, frames, patch):
        patch.set_data(frames[num])
        return patch,

    def plot_animation(frames:list, save_path:str, title:str, repeat=False, interval=50):
        fig = plt.figure()
        patch = plt.imshow(frames[0])
        plt.axis('off')
        plt.title(title, fontsize=10)
        animation = anim.FuncAnimation(
            fig, update_scene, fargs=(frames, patch),
            frames=len(frames), repeat=repeat, interval=interval)
        animation.save(save_path, writer="pillow", fps=20)
        return animation

    def show_one_episode(action_sampler:Callable, save_path:str, repeat=False):
        frames = []
        state, info = env.reset()
        phi = torch.cat([
                    prepro_state:=prepro(state, DEVICE),
                    torch.zeros_like(prepro_state),
                    torch.zeros_like(prepro_state),
                    torch.zeros_like(prepro_state)
                ]).unsqueeze(0) # (1, 4, 80, 80)
        sum_rewards = 0
        with torch.no_grad():
            for step in count(0):
                frames.append(env.render())
                action = action_sampler(phi)
                state, reward, done, truncated, info = env.step(action)
                phi = torch.cat([prepro(state).unsqueeze(0), phi[:, :-1]], dim=1)
                sum_rewards += reward
                if done or truncated:
                    print(f"|| done at step: {step+1} ||")
                    print(f"|| sum_rewards: {sum_rewards} ||")
                    break
        env.close()
        title = "Our trained agent on the right playing with a hardcoded agent on the left side"
        return plot_animation(frames, save_path, title=title, repeat=repeat)


    dqn = get_model(1, 4, 80, 80, log=False, fan_in=4, fan_out=int(env.action_space.n))
    dqn.eval()
    dqn.compile()

    @torch.no_grad()
    def sample_action(obs:torch.Tensor):
        return dqn(obs).squeeze(0).argmax()

    dqn.load_state_dict(torch.load(MODEL_WEI_PATH, map_location=DEVICE, weights_only=True))

    show_one_episode(sample_action, "images/dqn_pong.gif", repeat=False)