from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator

from .basic_tools import process_dataset, CustomDataset, stats
from config import TASK_TYPE, get_type, TASK_PATHS_MAPPING, MAX_LENGTH_MAPPING


def get_dataloader(
    accelerator: Accelerator,
    tokenizer,
    config,
    collect_stats=True,
    only_val=False,
    only_test=False,
):
    train_dataset = []
    val_dataset = []
    gt_key = None
    # 1. Load dataset
    task_type: TASK_TYPE = get_type(TASK_TYPE, config.get("task_name", None))
    if task_type not in TASK_PATHS_MAPPING.keys():
        raise NotImplementedError(
            f"No path found in TASK_PATHS_MAPPING for {task_type.__name__}, please specify one."
        )
    if task_type not in MAX_LENGTH_MAPPING.keys():
        raise NotImplementedError(
            f"No path found in MAX_LENGTH_MAPPING for {task_type.__name__}, please specify one."
        )

    if task_type in (TASK_TYPE.COMMONSENSE170K,) and config.data.batch_size > 1:
        raise ValueError("Batch size must be 1 for commonsense_170k dataset.")
    
    # Get the mapping for the current task
    task_info = TASK_PATHS_MAPPING[task_type]
    
    # Determine which split to load
    split_to_load = "test" if only_test else "train"

    fixed_train_file = config.data.get("train_file", None) if hasattr(config, "data") else None
    fixed_val_file = config.data.get("val_file", None) if hasattr(config, "data") else None

    if fixed_train_file and (fixed_val_file or only_test):
        data_files = {"train": fixed_train_file}
        if fixed_val_file:
            data_files["val"] = fixed_val_file
        dataset = load_dataset("json", data_files=data_files, split="train")
        dataset_val_fixed = load_dataset("json", data_files=data_files, split="val") if fixed_val_file else None

    elif "data_files" in task_info:

        # Check if the split we want is defined
        if split_to_load not in task_info["data_files"]:
            raise ValueError(
                f"Split '{split_to_load}' is not defined in data_files for {task_type}. "
                f"Available splits: {list(task_info['data_files'].keys())}"
            )
            
        dataset = load_dataset(
            "json",  # Specify the json loader
            data_files=task_info["data_files"], # Pass the file paths
            split=split_to_load                 # Select the split
        )
        # --- END of new logic ---
    
    else:
        # --- OLD LOGIC for Hub datasets ---
        dataset = load_dataset(*(task_info["path"]), split=split_to_load)

    # The rest of your code remains the same
    q_key = task_info["q_key"]
    a_key = task_info["a_key"]
    gt_key = task_info.get("gt_key", None)

    # 2. Train/val split
    if not only_test:
        if fixed_train_file and fixed_val_file:
            dataset_train = dataset
            dataset_val = dataset_val_fixed
        else:
            split_dataset = dataset.train_test_split(
                test_size=config.data.val_split_size,
                seed=config.data.val_split_seed,  # reproducibility
                shuffle=True,
            )
            dataset_train = split_dataset["train"]
            dataset_val = split_dataset["test"]

    max_length: int = MAX_LENGTH_MAPPING[task_type]

    # Initialize stats dictionary
    if collect_stats and (not only_test):
        stats_dict = {
            "train": {"q_lens": [], "a_lens": [], "total_lens": []},
            "val": {"q_lens": [], "a_lens": [], "total_lens": []},
        }
    elif collect_stats and only_test:
        stats_dict = {
            "test": {"q_lens": [], "a_lens": [], "total_lens": []},
        }
    else:
        stats_dict = None

    # 3. Process train/val datasets
    if not only_test:
        if not only_val:
            train_dataset = process_dataset(
                dataset_train,
                "Processing Train Data",
                "train",
                accelerator,
                tokenizer,
                max_length,
                collect_stats,
                stats_dict,
                q_key,
                a_key,
                gt_key,
                task_type,
            )
            train_dataloader = DataLoader(
                CustomDataset(train_dataset),
                batch_size=config.data.batch_size,
                # num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
            if accelerator.is_main_process and collect_stats:
                print("=" * 60)
                print("📊 Train Dataset Statistics")
                print(f"Train size: {len(train_dataset)}")
                split_name = "train"
                print(f"--- {split_name.upper()} ---")
                print(stats("Question Token Length", stats_dict[split_name]["q_lens"]))
                print(stats("Answer Token Length", stats_dict[split_name]["a_lens"]))
                print(stats("Total Token Length", stats_dict[split_name]["total_lens"]))
                print("=" * 60)
        val_dataset = process_dataset(
            dataset_val,
            "Processing Val Data",
            "val",
            accelerator,
            tokenizer,
            max_length,
            collect_stats,
            stats_dict,
            q_key,
            a_key,
            gt_key,
            task_type,
        )

        val_dataloader = DataLoader(
            CustomDataset(val_dataset),
            batch_size=1,
            num_workers=0,
            shuffle=False,  # Validation set is not shuffled
            pin_memory=True,
        )

        # 5. Print dataset statistics (main process only)
        if accelerator.is_main_process and collect_stats:
            print("=" * 60)
            print("📊 Val Dataset Statistics")
            print(f"Val size: {len(val_dataset)}")
            split_name = "val"
            print(f"--- {split_name.upper()} ---")
            print(stats("Question Token Length", stats_dict[split_name]["q_lens"]))
            print(stats("Answer Token Length", stats_dict[split_name]["a_lens"]))
            print(stats("Total Token Length", stats_dict[split_name]["total_lens"]))
            print("=" * 60)
        if not only_val:
            return train_dataloader, val_dataloader
        else:
            return val_dataloader
    elif only_test:
        test_dataset = process_dataset(
            dataset,
            "Processing Test Data",
            "test",
            accelerator,
            tokenizer,
            max_length,
            collect_stats,
            stats_dict,
            q_key,
            a_key,
            gt_key,
            task_type,
        )

        test_dataloader = DataLoader(
            CustomDataset(test_dataset),
            batch_size=config.data.batch_size,
            # num_workers=0,
            shuffle=False,  # Test set is not shuffled
            pin_memory=True,
        )

        # 5. Print dataset statistics (main process only)
        if accelerator.is_main_process and collect_stats:
            print("=" * 60)
            print("📊 Test Dataset Statistics")
            print(f"Test size: {len(test_dataset)}")
            split_name = "test"
            print(f"--- {split_name.upper()} ---")
            print(stats("Question Token Length", stats_dict[split_name]["q_lens"]))
            print(stats("Answer Token Length", stats_dict[split_name]["a_lens"]))
            print(stats("Total Token Length", stats_dict[split_name]["total_lens"]))
            print("=" * 60)

        return test_dataloader
