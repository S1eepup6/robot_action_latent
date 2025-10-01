import torch
import einops
import numpy as np
from pathlib import Path
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
from utils.libero_dataset_core import TrajectoryDataset
# from libero_dataset_core import TrajectoryDataset
from transformers import BertTokenizer, BertModel
import re
from tqdm import tqdm

### Compute the bert embedding of the task description
def get_task_embedding(task_name):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = tokenizer.encode(task_name, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])
    model = BertModel.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        outputs = model(input_tensor)
    last_hidden_states = outputs[0]
    sentence_embedding = last_hidden_states[:, 0, :]
    return sentence_embedding

def extract_task_information(file_name, libero_path):
    """
    Extracts task information from the given file name.
    """
    # Regular expression pattern to extract the task name
    pattern = r'{}/((.+)_(.+))_demo\.hdf5'.format(libero_path)

    # Extracting the task name
    match = re.search(pattern, file_name)
    
    task_embedding = get_task_embedding(match.group(3).lower().replace("_", " "))
    print(match.group(3).lower().replace("_", " "))
    return match.group(1).lower() if match else None, task_embedding


class LiberoGoalDataset(TrajectoryDataset):
    # data structure:
    # libero_goal
    #      task_name
    #          demo_{i}
    #              agentview_image.mp4
    #              robot0_eye_in_hand_image.mp4
    #              robot0_joint_pos.npy
    #              robot0_eef.npy
    #              robot0_gripper_qpos.npy
    #              object_states.npy
    #              actions.npy
    def __init__(self, 
                 data_directory = "/data/libero/libero_dataset",
                 subset_fraction: Optional[float] = None):
        self.dir = Path(data_directory) / "libero_goal"
        self.task_names = list(self.dir.iterdir())
        self.task_names.sort()
        self.demos = []
        self.goals = []
        for task_name in tqdm(self.task_names):
            task_id = str(task_name).split('/')[-1]
            self.goals.append(get_task_embedding(task_id.lower().replace("_", " ")))
            self.demos += list(task_name.iterdir())

        self.subset_fraction = subset_fraction
        if self.subset_fraction:
            assert 50 % self.subset_fraction == 0
            # n = int(len(self.demos) * self.subset_fraction)
            self.demos = self.demos[::self.subset_fraction]

        # prefetch all npy data
        self.joint_pos = []
        self.eef = []
        self.gripper_qpos = []
        self.object_states = []
        self.states = []
        self.actions = []
        for demo in self.demos:
            self.joint_pos.append(np.load(demo / "robot0_joint_pos.npy"))
            self.eef.append(np.load(demo / "robot0_eef.npy"))
            self.gripper_qpos.append(np.load(demo / "robot0_gripper_pos.npy"))
            self.object_states.append(np.load(demo / "object_states.npy"))
            state = np.concatenate(
                [
                    self.joint_pos[-1],
                    self.eef[-1],
                    self.gripper_qpos[-1],
                    self.object_states[-1],
                ],
                axis=1,
            )
            act = np.load(demo / "actions.npy")
            self.states.append(torch.from_numpy(state))
            self.actions.append(torch.from_numpy(act))

        # pad state dimension to same length for linear probe diagnostics
        MAX_DIM = 128
        for i in range(len(self.states)):
            self.states[i] = torch.cat(
                [
                    self.states[i],
                    torch.zeros(
                        self.states[i].shape[0], MAX_DIM - self.states[i].shape[1]
                    ),
                ],
                dim=1,
            )
        # pad states and actions to the same time length
        self.states = pad_sequence(self.states, batch_first=True).float()
        self.actions = pad_sequence(self.actions, batch_first=True).float()

        # last frame goal
        # self.goals = None
        # goals = []
        # for i in range(10):
        #     last_obs, _, _ = self.get_frames(i, [-1])  # 1 V C H W
        #     goals.append(last_obs)
        # self.goals = goals

    def __len__(self):
        return len(self.demos)

    def get_frames(self, idx, frames):
        demo = self.demos[idx]
        agentview_obs = torch.load(
            str(demo / "agentview_image.pth"),
        )
        robotview_obs = torch.load(
            str(demo / "robot0_eye_in_hand_image.pth"),
        )
        agentview = agentview_obs[frames]
        robotview = robotview_obs[frames]
        obs = torch.stack([agentview, robotview], dim=1)
        obs = einops.rearrange(obs, "T V H W C -> T V C H W") / 255.0
        act = self.actions[idx][frames]

        # print(obs.shape)

        if self.goals is not None:
            task_idx = idx // int(50 // self.subset_fraction)
            goal = self.goals[task_idx]
            return obs, act, goal
        else:
            return obs, act, None

    def __getitem__(self, idx):
        return self.get_frames(idx, range(len(self.joint_pos[idx])))

    def get_seq_length(self, idx):
        return len(self.joint_pos[idx])

    def get_all_actions(self):
        actions = []
        for i in range(len(self.demos)):
            T = self.get_seq_length(i)
            actions.append(self.actions[i][:T])
        return torch.cat(actions, dim=0)


if __name__ == "__main__":
    
    from libero_dataset_core import get_train_val_sliced
    from torch.utils.data import DataLoader

    dataset = LiberoGoalDataset(subset_fraction=5)
    # dataset = LiberoGoalDataset()
    data_loader = DataLoader(dataset, shuffle=True, batch_size=64)
    
    kwargs = {
        "train_fraction": 0.999,
        "random_seed": 42,
        "window_size": 4+1,
        "future_conditional": False,
        "min_future_sep": 0,
        "future_seq_len": 0,
        "num_extra_predicted_actions": 0,
    }
    train_loader, test_loader = get_train_val_sliced(dataset, **kwargs)