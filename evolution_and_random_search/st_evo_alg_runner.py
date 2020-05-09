import gym

import numpy as np
import streamlit as st
import os
import pickle

from ARS_bc import augmentedRandomSearch

# from sac_openai_cuda import sac, OUNoise
# from sac_openai_cuda_nets import MLPActorCritic
from plot_episode_logs import plot_episode_logs


selected_alg = st.selectbox("Select an algorithm", ["ARS"],)

algs_by_name = {"ARS": augmentedRandomSearch}

algorithm = algs_by_name[selected_alg]

selected_env = st.selectbox(
    "Select an environment",
    [
        "Pendulum-v0",
        "BipedalWalker-v2",
        "BipedalWalkerHardcore-v2",
        "MountainCarContinuous-v0",
        "LunarLanderContinuous-v2",
    ],
)


def env_fn():
    return gym.make(selected_env)


episodes_per_epoch = st.number_input(
    label="episodes per epoch", min_value=0, value=4000, step=1, format="%.0d",
)

epochs = st.number_input(label="epochs", min_value=0, value=100, step=1, format="%.0d",)


directions_per_episode = st.number_input(
    label="perturbations per episode", min_value=0, value=16, step=1, format="%.0d",
)

# start_steps = st.number_input(
#     label="random steps at start", min_value=0, value=10000, step=1, format="%.0d",
# )


train_fresh_model = st.button("train a fresh model")
fresh_train_plot = st.empty()


def model_trainer_fig_update_handler(fig):
    fresh_train_plot.pyplot(fig)


if train_fresh_model:
    ac, step_log, episode_log = algorithm(
        env_fn,
        env_name=selected_env,
        episodes_per_epoch=episodes_per_epoch,
        epochs=epochs,
        epoch_plot_fig_handler=model_trainer_fig_update_handler,
        directions_per_episode=directions_per_episode,
        max_steps_per_episode=500,
    )


def run_selector(folder_path="."):
    dir_names = os.listdir(folder_path)
    dirs_by_dir_labels = {
        f"{dirname}"
        f' ({len(os.listdir(f"{folder_path}/{dirname}"))}'
        " checkpoints)": dirname
        for dirname in dir_names
    }
    selected_label = st.selectbox("Select a run", sorted(list(dirs_by_dir_labels)))
    selected_dir = dirs_by_dir_labels[selected_label]
    return os.path.join(folder_path, selected_dir)


run_dirname = run_selector(f"model_runs/{selected_alg}/{selected_env}")
st.write(f"You selected run `{run_dirname}`")


with open(run_dirname + "/log.pkl", "rb") as pickle_file:
    # print(pickle_file)
    log_info = pickle.load(pickle_file)
# print(list(log_info))
st.write(log_info["run params"])

episode_log = log_info["episode log"]
# step_log = log_info["step log"]

fig, plt_update_fn = plot_episode_logs(episode_log)
plt_update_fn()
st.pyplot(fig)


def checkpoint_selector(folder_path="."):
    filenames = [f for f in sorted(os.listdir(folder_path)) if f[:5] == "model"]
    selected_filename = st.selectbox("Select a file", sorted(filenames))
    return os.path.join(folder_path, selected_filename)


checkpoint_filename = checkpoint_selector(run_dirname)
st.write("You selected model checkpoint `%s`" % checkpoint_filename)


def load_and_run_model(checkpoint_filename, num_runs=6):

    env = gym.make(selected_env)

    pi_weights = np.load(checkpoint_filename)

    def act(state, pi_weights):
        return np.matmul(pi_weights, state.reshape(-1, 1))

    for i in range(10):
        print(f"showing {i} of {num_runs} runs")
        state = env.reset()
        for j in range(300):
            env.render()
            action = act(state, pi_weights)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break
    env.close()


run_trained_model = st.button("run trained model (in new window)")
if run_trained_model:
    load_and_run_model(checkpoint_filename)

trained_model = np.array(
    [
        [
            5.664643480891290439e-02,
            3.442509575903311414e-01,
            1.422129812821736616e-01,
            -2.820029842249014429e-01,
            -3.148828431430211738e-01,
            -1.904522359623921079e-01,
            2.617115452072476395e-02,
            -5.148315371935829193e-02,
            -7.437201162861399595e-02,
            3.214536795746698872e-02,
            -3.221106312323215315e-01,
            -6.093415372231254096e-02,
            1.283266775976633634e-01,
            1.055026785281292928e-01,
            -1.726894675034008064e-01,
            -8.192990040422867826e-02,
            2.868107672037503025e-02,
            -1.253313255786948077e-01,
            -5.000716171885470906e-01,
            1.345367984943873774e-01,
            -2.504210436577612997e-01,
            -1.531160085684604016e-02,
            1.748923739327276905e-01,
            1.405171249513042508e-01,
        ],
        [
            2.130214643858053813e-01,
            3.481272568064924799e-01,
            -3.156490159065780676e-01,
            -3.208335634678360915e-01,
            1.937820123350550705e-01,
            -2.390305608493398237e-01,
            -4.892406816658725899e-01,
            -2.761943788722150406e-01,
            5.211579481891609467e-02,
            5.203502882570690824e-01,
            4.239234030645436246e-01,
            3.108966179897078019e-01,
            -3.121349276110329529e-01,
            -6.150488519516657086e-03,
            -9.902088314983129025e-02,
            -3.354608847260510118e-01,
            5.253241087916174501e-02,
            -1.585220858956457035e-01,
            -1.693884445808936379e-01,
            -3.072411465882980974e-01,
            -2.466220718185950511e-01,
            -1.925877653710476589e-01,
            1.150343328927397671e-02,
            1.546064834362041174e-01,
        ],
        [
            3.744776386395396450e-01,
            5.502060910887280470e-02,
            4.918783039943843555e-02,
            -2.083996004326419960e-01,
            3.155261325025408847e-02,
            -2.896292683920498878e-01,
            -1.993679288973738417e-01,
            7.540726649987453123e-02,
            -2.960566215720621397e-01,
            -3.161118894561018577e-01,
            -6.033683171418132568e-02,
            -6.200127726523580601e-03,
            -7.423534652720473193e-02,
            1.160514050226940652e-02,
            2.067201777276269081e-01,
            -4.375545858996230525e-02,
            -2.771206250455741579e-02,
            3.149302176317955726e-01,
            4.189467052853208529e-01,
            -4.273644350507962292e-02,
            -5.590379306824899697e-02,
            4.959897504024527448e-01,
            1.942708869529923554e-01,
            3.108959201978611989e-02,
        ],
        [
            1.274355387661604277e-01,
            1.917181072665868624e-01,
            4.624036494822651028e-02,
            -3.249165333028166680e-01,
            4.636644887833415996e-02,
            2.075135169488562803e-01,
            1.946228138233243310e-01,
            5.821126277878674316e-02,
            5.015622432940998118e-03,
            -3.086118108968032708e-02,
            -2.030495039296549842e-01,
            -4.697052923144355296e-01,
            -7.490075249547256675e-02,
            2.940938460480205241e-02,
            -3.861245499636513367e-02,
            -9.562870044925621260e-02,
            -1.603265734106108509e-01,
            -8.533314353063035929e-02,
            -7.449787439054850191e-02,
            -1.683713859059103868e-01,
            3.850237059515051151e-03,
            1.372927564296815839e-01,
            -1.816074936948310348e-01,
            1.011795535024103271e-01,
        ],
    ]
)

obs_mean = np.array(
    [
        3.671218105051134994e-01,
        4.385550343327458797e-04,
        2.987120773647629068e-01,
        -3.471893844289233779e-03,
        -5.872217981797950292e-01,
        -1.288406245368736193e-02,
        2.350884310824232215e-01,
        -6.143080717325245799e-02,
        4.920221375566185551e-01,
        7.067761684920256249e-01,
        -4.261759607812246227e-03,
        -4.526924528129612768e-01,
        -2.569230577349456909e-02,
        3.358401757089036765e-01,
        3.900445584930448129e-01,
        3.944451638865039689e-01,
        4.082783868578218067e-01,
        4.332603021302398605e-01,
        4.728678295053651559e-01,
        5.336852450839811857e-01,
        6.286368556808481500e-01,
        7.852642697420150197e-01,
        9.784117012675991321e-01,
        9.996836127952202222e-01,
    ]
)

obs_var = np.array(
    [
        3.806674427142537481e-02,
        1.000000000000000021e-02,
        1.283332999564747341e-02,
        1.000000000000000021e-02,
        5.420818393616537445e-02,
        4.412358723081376555e-01,
        1.601511983168304454e-01,
        4.681866999647312233e-01,
        2.499363537108744149e-01,
        1.677527945672853960e-01,
        5.670493657650788055e-01,
        5.180750170077494388e-02,
        2.780057174669389730e-01,
        2.230515520887144942e-01,
        1.000000000000000021e-02,
        1.000000000000000021e-02,
        1.000000000000000021e-02,
        1.000000000000000021e-02,
        1.000000000000000021e-02,
        1.000000000000000021e-02,
        1.000000000000000021e-02,
        1.000000000000000021e-02,
        1.000000000000000021e-02,
        1.000000000000000021e-02,
    ]
)


def normalize(inputs):
    # obs_mean = self.mean
    obs_std = np.sqrt(obs_var)
    return (inputs - obs_mean) / obs_std


def run_pre_trained_biped_model(num_runs=6):

    env = gym.make("BipedalWalker-v2")

    pi_weights = trained_model

    def act(state, pi_weights):
        return np.matmul(pi_weights, state.reshape(-1, 1))

    for i in range(10):
        print(f"showing {i} of {num_runs} runs")
        state = env.reset()
        for j in range(1000):
            env.render()
            action = act(normalize(state), pi_weights)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break
    env.close()


run_trained_internet_model = st.button("run biped model from internet (in new window)")
if run_trained_internet_model:
    run_pre_trained_biped_model()


# def newAcFromOld(old_net):
#     def constructor(*args, **kwargs):
#         new_net = MLPActorCritic(*args, **kwargs)
#         new_net.pi = old_net.pi
#         new_net.q1 = old_net.q1
#         new_net.q2 = old_net.q2
#         new_net.device = old_net.device
#         return new_net

#     return constructor


# train_more = st.button("keep training")
# if train_more:
#     env = gym.make("BipedalWalker-v2")
#     actor_net = MLPActorCritic(env.observation_space, env.action_space)

#     actor_net.load_state_dict(torch.load(checkpoint_filename))
#     actor_net.to(device)

#     st_more_training_plot = st.empty()

#     actor_net2 = sac(
#         env_fn,
#         actor_critic=newAcFromOld(actor_net),
#         start_steps=0,
#         episodes_per_epoch=500,
#         # max_ep_len=100,
#         update_every=1,
#         epochs=200,
#         use_logger=False,
#         alpha=0.2,
#         lr=0.0005,
#         # batch_size=256,
#         # lr=7e-3,
#         add_noise=True,
#         device=device,
#         st_plot_fn=st_more_training_plot.pyplot,
#     )
