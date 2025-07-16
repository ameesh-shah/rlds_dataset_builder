from collections import defaultdict
from copy import deepcopy
from glob import glob
import json
import os
from typing import Any, Iterator, Tuple

import cv2
import h5py
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds

camera_type_dict = {
    "hand_camera_id": 0,
    "varied_camera_1_id": 1,
    "varied_camera_2_id": 1,
}

camera_type_to_string_dict = {
    0: "hand_camera",
    1: "varied_camera",
    2: "fixed_camera",
}


def get_camera_type(cam_id):
    if cam_id not in camera_type_dict:
        return None
    type_int = camera_type_dict[cam_id]
    return camera_type_to_string_dict[type_int]


class MP4Reader:
    def __init__(self, filepath, serial_number):
        # Save Parameters #
        self.serial_number = serial_number
        self._index = 0

        # Open Video Reader #
        self._mp4_reader = cv2.VideoCapture(filepath)
        if not self._mp4_reader.isOpened():
            raise RuntimeError("Corrupted MP4 File")

    def set_reading_parameters(
        self,
        image=True,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        # Save Parameters #
        self.image = image
        self.concatenate_images = concatenate_images
        self.resolution = resolution
        self.resize_func = cv2.resize
        self.skip_reading = not image
        if self.skip_reading:
            return

    def get_frame_resolution(self):
        width = self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        return (width, height)

    def get_frame_count(self):
        if self.skip_reading:
            return 0
        frame_count = int(self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        return frame_count

    def set_frame_index(self, index):
        if self.skip_reading:
            return

        if index < self._index:
            self._mp4_reader.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            self._index = index

        while self._index < index:
            self.read_camera(ignore_data=True)

    def _process_frame(self, frame):
        frame = deepcopy(frame)
        if self.resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resolution)

    ## YY: edited this method, no more single_width nonsense, just set serial_number_left and serial_number_right to the same image
    def read_camera(self, ignore_data=False, correct_timestamp=None):
        # Skip if Read Unnecesary #
        if self.skip_reading:
            return {}

        # Read Camera #
        success, frame = self._mp4_reader.read()

        self._index += 1
        if not success:
            return None
        if ignore_data:
            return None

        # Return Data #
        data_dict = {}

        if self.concatenate_images:
            data_dict["image"] = {self.serial_number: self._process_frame(frame)}
        else:
            data_dict["image"] = {
                self.serial_number + "_left": self._process_frame(frame[:, :, :]),
                self.serial_number + "_right": self._process_frame(frame[:, :, :]),
            }
        return data_dict

    def disable_camera(self):
        if hasattr(self, "_mp4_reader"):
            self._mp4_reader.release()


class RecordedMultiCameraWrapper:
    def __init__(self, recording_folderpath, camera_kwargs={}):
        # Save Camera Info #
        self.camera_kwargs = camera_kwargs

        # Open Camera Readers #
        mp4_filepaths = glob(recording_folderpath + "/*.mp4")
        svo_filepaths = glob(recording_folderpath + "/*.svo")
        all_filepaths = svo_filepaths + mp4_filepaths

        self.camera_dict = {}
        for f in all_filepaths:
            serial_number = f.split("/")[-1][:-4]
            cam_type = get_camera_type(serial_number)
            camera_kwargs.get(cam_type, {})

            if f.endswith(".mp4"):
                reader = MP4Reader
            else:
                raise ValueError

            ### TODO hack again, swapping out serial numbers for 1 troublesome case
            if serial_number == "24514023":
                self.camera_dict["20521388"] = reader(f, "20521388")
            else:
                self.camera_dict[serial_number] = reader(f, serial_number)

    def read_cameras(self, index=None, camera_type_dict=None, timestamp_dict=None):
        camera_type_dict = camera_type_dict or {}
        timestamp_dict = timestamp_dict or {}
        full_obs_dict = defaultdict(dict)

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        # random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            cam_type = camera_type_dict[cam_id]
            curr_cam_kwargs = self.camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

            timestamp = timestamp_dict.get(cam_id + "_frame_received", None)
            if index is not None:
                self.camera_dict[cam_id].set_frame_index(index)

            data_dict = self.camera_dict[cam_id].read_camera(correct_timestamp=timestamp)

            # Process Returned Data #
            if data_dict is None:
                return None
            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])

        return full_obs_dict


def get_hdf5_length(hdf5_file, keys_to_ignore=None):
    keys_to_ignore = keys_to_ignore or []
    length = None

    for key in hdf5_file.keys():
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            curr_length = get_hdf5_length(curr_data, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            curr_length = len(curr_data)
        else:
            raise ValueError

        if length is None:
            length = curr_length
        assert curr_length == length

    return length


def load_hdf5_to_dict(hdf5_file, index, keys_to_ignore=None):
    data_dict = {}
    keys_to_ignore = keys_to_ignore or []
    for key in hdf5_file:
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            data_dict[key] = load_hdf5_to_dict(curr_data, index, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            data_dict[key] = curr_data[index]
        else:
            raise ValueError

    return data_dict


class TrajectoryReader:
    def __init__(self, filepath, read_images=True):
        self._hdf5_file = h5py.File(filepath, "r")
        is_video_folder = "observations/videos" in self._hdf5_file
        self._read_images = read_images and is_video_folder
        self._length = get_hdf5_length(self._hdf5_file)
        self._video_readers = {}
        self._index = 0

    def length(self):
        return self._length

    def read_timestep(self, index=None, keys_to_ignore=None):
        keys_to_ignore = keys_to_ignore or []
        # Make Sure We Read Within Range #
        if index is None:
            index = self._index
        else:
            assert not self._read_images
            self._index = index
        assert index < self._length

        # Load Low Dimensional Data #
        keys_to_ignore = [*keys_to_ignore.copy(), "videos"]
        timestep = load_hdf5_to_dict(self._hdf5_file, self._index, keys_to_ignore=keys_to_ignore)

        # Increment Read Index #
        self._index += 1

        # Return Timestep #
        return timestep

    def close(self):
        self._hdf5_file.close()


def crawler(dirname, filter_func=None):
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    traj_files = [f.path for f in os.scandir(dirname) if (f.is_file() and "trajectory.h5" in f.path)]

    if len(traj_files):
        # Only Save Desired Data #
        if filter_func is None:
            use_data = True
        else:
            hdf5_file = h5py.File(traj_files[0], "r")
            use_data = filter_func(hdf5_file.attrs)
            hdf5_file.close()

        if use_data:
            return [dirname]

    all_folderpaths = []
    for child_dirname in subfolders:
        child_paths = crawler(child_dirname, filter_func=filter_func)
        all_folderpaths.extend(child_paths)

    return all_folderpaths


def load_trajectory(
    filepath,
    metadata_filepath,
    read_cameras=True,
    recording_folderpath=None,
    camera_kwargs=None,
    remove_skipped_steps=False,
    num_samples_per_traj=None,
    num_samples_per_traj_coeff=1.5,
):
    read_hdf5_images = read_cameras and (recording_folderpath is None)
    read_recording_folderpath = read_cameras and (recording_folderpath is not None)
    camera_kwargs = camera_kwargs or {}

    traj_reader = TrajectoryReader(filepath, read_images=read_hdf5_images)
    metadata = json.load(open(metadata_filepath, "r"))
    if read_recording_folderpath:
        camera_reader = RecordedMultiCameraWrapper(recording_folderpath, camera_kwargs)

    task_instruction = metadata["current_task"]

    horizon = traj_reader.length()
    timestep_list = []

    # Choose Timesteps To Save #
    if num_samples_per_traj:
        num_to_save = num_samples_per_traj
        if remove_skipped_steps:
            num_to_save = int(num_to_save * num_samples_per_traj_coeff)
        max_size = min(num_to_save, horizon)
        indices_to_save = np.sort(np.random.choice(horizon, size=max_size, replace=False))
    else:
        indices_to_save = np.arange(horizon)

    # Iterate Over Trajectory #
    for i in indices_to_save:
        # Get HDF5 Data #
        timestep = traj_reader.read_timestep(index=i)

        # If Applicable, Get Recorded Data #
        if read_recording_folderpath:
            timestamp_dict = timestep["observation"]["timestamp"]["cameras"]
            camera_type_dict = {
                k: camera_type_to_string_dict[v] for k, v in timestep["observation"]["camera_type"].items()
            }

            ## TODO: temp hack, for some reason timestamp_dict is differing, has 24514023... instead of 20521388_... (24514023 is the correct SN, but all other data has 20521388, so just changing timestep_dict to lineup with everyone else here)
            new_timestamp_dict = {}
            for k, v in timestamp_dict.items():
                if k.startswith("24514023"):
                    new_key = "20521388" + k[8:]
                    new_timestamp_dict[new_key] = v
                else:
                    new_timestamp_dict[k] = v
            timestamp_dict = new_timestamp_dict

            camera_obs = camera_reader.read_cameras(
                index=i, camera_type_dict=camera_type_dict, timestamp_dict=timestamp_dict
            )
            camera_failed = camera_obs is None

            # Add Data To Timestep If Successful #
            if camera_failed:
                continue  # Should this be a continue, or a break? It was originally a break but that would end the whole loop.
            else:
                timestep["observation"].update(camera_obs)

        # Filter Steps #
        step_skipped = not timestep["observation"]["controller_info"].get("movement_enabled", True)
        delete_skipped_step = step_skipped and remove_skipped_steps

        # Save Filtered Timesteps #
        if delete_skipped_step:
            del timestep
        else:
            timestep_list.append(timestep)

    # Remove Extra Transitions #
    timestep_list = np.array(timestep_list)
    if (num_samples_per_traj is not None) and (len(timestep_list) > num_samples_per_traj):
        ind_to_keep = np.random.choice(len(timestep_list), size=num_samples_per_traj, replace=False)
        timestep_list = timestep_list[ind_to_keep]

    # Close Readers #
    traj_reader.close()

    # Return Data #
    return timestep_list, task_instruction


class GenericDroidFinetuningSet(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, raw_data_path, *args, **kwargs):
        self.raw_data_path = raw_data_path
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "exterior_image_1": tfds.features.Image(
                                        shape=(180, 320, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Exterior camera 1 left viewpoint",
                                    ),
                                    "exterior_image_2": tfds.features.Image(
                                        shape=(180, 320, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Exterior camera 2 left viewpoint",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(180, 320, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB left viewpoint",
                                    ),
                                    "gripper_position": tfds.features.Tensor(
                                        shape=(1,),
                                        dtype=np.float64,
                                        doc="Gripper position statae",
                                    ),
                                    "joint_position": tfds.features.Tensor(
                                        shape=(7,), dtype=np.float64, doc="Joint position state"
                                    ),
                                }
                            ),
                            "actions": tfds.features.Tensor(
                                shape=(8,),
                                dtype=np.float64,
                                doc="Robot action, consists of [7x joint velocities, \
                            1x gripper position].",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32, doc="Discount if provided, default to 1."
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32, doc="Reward if provided, 1 on final step for demos."
                            ),
                            "is_first": tfds.features.Scalar(dtype=np.bool_, doc="True on first step of the episode."),
                            "is_last": tfds.features.Scalar(dtype=np.bool_, doc="True on last step of the episode."),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(doc="Language Instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(doc="Path to the original data file."),
                            "recording_folderpath": tfds.features.Text(doc="Path to the folder of recordings."),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(path=self.raw_data_path),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _resize_and_encode(image, size):
            if image.shape[-1] == 3:
                image = image[..., ::-1]
                image = Image.fromarray(image)
                return np.array(image.resize(size, resample=Image.BILINEAR))
            return None

        def _parse_example(episode_path):
            FRAMESKIP = 1
            IMAGE_SIZE = (320, 180)

            h5_filepath = os.path.join(episode_path, "trajectory.h5")
            recording_folderpath = os.path.join(episode_path, "recordings", "MP4")
            metadata_filepath = glob(os.path.join(episode_path, "metadata*.json"))[0]

            traj, instruction = load_trajectory(
                h5_filepath, metadata_filepath, recording_folderpath=recording_folderpath
            )
            data = traj[::FRAMESKIP]

            assert all(t.keys() == data[0].keys() for t in data)
            for t in range(len(data)):
                for key in data[0]["observation"]["image"].keys():
                    data[t]["observation"]["image"][key] = _resize_and_encode(
                        data[t]["observation"]["image"][key], IMAGE_SIZE
                    )

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                obs = step["observation"]
                action = step["action"]
                camera_type_dict = obs["camera_type"]
                wrist_ids = [k for k, v in camera_type_dict.items() if v == 0]
                exterior_ids = [k for k, v in camera_type_dict.items() if v != 0]

                episode.append(
                    {
                        "observation": {
                            "exterior_image_1": obs["image"][f"{exterior_ids[0]}_left"],
                            "exterior_image_2": obs["image"][f"{exterior_ids[1]}_left"],
                            "wrist_image": obs["image"][f"{wrist_ids[0]}_left"],
                            "joint_position": obs["robot_state"]["joint_positions"],
                            "gripper_position": np.array([obs["robot_state"]["gripper_position"]]),
                        },
                        "actions": np.concatenate((action["joint_velocity"], [action["gripper_position"]])),
                        "discount": 1.0,
                        "reward": float(i == (len(data) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(data) - 1),
                        "is_terminal": i == (len(data) - 1),
                        "language_instruction": instruction,
                    }
                )
            # create output data sample
            sample = {
                "steps": episode,
                "episode_metadata": {"file_path": h5_filepath, "recording_folderpath": recording_folderpath},
            }
            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = crawler(path)
        episode_paths = [
            p for p in episode_paths if os.path.exists(p + "/trajectory.h5") and os.path.exists(p + "/recordings/MP4")
        ]

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
