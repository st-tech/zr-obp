import argparse

from obp.dataset import (
    logistic_reward_function,
    linear_behavior_policy_logit,
    SyntheticSlateBanditDataset,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run slate dataset.")
    parser.add_argument(
        "--n_unique_action", type=int, default=10, help="number of unique actions."
    )
    parser.add_argument(
        "--len_list", type=int, default=3, help="number of item positions."
    )
    parser.add_argument("--n_rounds", type=int, default=100, help="number of slates.")
    parser.add_argument(
        "--clip_logit_value",
        type=float,
        default=None,
        help="a float parameter to clip logit value.",
    )
    parser.add_argument(
        "--is_factorizable",
        type=bool,
        default=False,
        help="a boolean parameter whether to use factorizable evaluation policy.",
    )
    parser.add_argument(
        "--return_pscore_item_position",
        type=bool,
        default=True,
        help="a boolean parameter whether `pscore_item_position` is returned or not",
    )
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=args.n_unique_action,
        dim_context=5,
        len_list=args.len_list,
        base_reward_function=logistic_reward_function,
        behavior_policy_function=linear_behavior_policy_logit,
        reward_type="binary",
        reward_structure="cascade_additive",
        click_model="cascade",
        random_state=12345,
        is_factorizable=args.is_factorizable,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=args.n_rounds,
        return_pscore_item_position=args.return_pscore_item_position,
        clip_logit_value=args.clip_logit_value,
    )
