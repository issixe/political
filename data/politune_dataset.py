from torchtune.datasets._preference import PreferenceDataset
from torchtune.data import Message
from datasets import load_dataset
from typing import Any, Dict, Mapping
import torchtune.data as data_module


class PoliticalPreferenceDataset(PreferenceDataset):
    pass


def politune_dataset(tokenizer, source, split="train", **kwargs):
    from torchtune.datasets import preference_dataset

    class PlainTextToMessages:
        def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
            chosen_messages = [
                Message(role="user", content=sample["prompt"], masked=True, eot=True),
                Message(role="assistant", content=sample["chosen"], masked=False, eot=True),
            ]
            rejected_messages = [
                Message(role="user", content=sample["prompt"], masked=True, eot=True),
                Message(role="assistant", content=sample["rejected"], masked=False, eot=True),
            ]
            return {"chosen": chosen_messages, "rejected": rejected_messages}

    from torchtune.datasets._preference import PreferenceDataset
    return PreferenceDataset(
        tokenizer=tokenizer,
        source=source,
        message_transform=PlainTextToMessages(),
        split=split,
    )
