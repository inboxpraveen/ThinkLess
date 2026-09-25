# Sources and licenses of the task files

The files in this directory are samples of public datasets, redistributed under
their licenses. Each was changed in the same ways: rows were sampled with a fixed
seed, labels were renamed to readable text (underscores become spaces), and
records were reformatted as JSON Lines with an `id`, a `state`, a `label` and a
`source` pointing back to the original row. Task-specific changes are listed
below. The ThinkLess license does not apply to these files; each keeps its own.

## banking77

- Source: https://github.com/PolyAI-LDN/task-specific-datasets (revision `57ec275d8078af65b7731c2a98be812d844a6d6b`)
- License: CC BY 4.0
- Citation: Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders, 2020 (PolyAI)
- Changes: 77 intents. Test rows from the test split, calibration rows from the train split.

## clinc150

- Source: https://huggingface.co/datasets/clinc/clinc_oos (revision `155b9c710419136e17307b80d0a13e68cd46b4ec`)
- License: CC BY 3.0
- Citation: Larson et al., An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction, 2019
- Changes: The 'out of scope' option covers requests outside every intent; about one row in five.

## massive

- Source: https://huggingface.co/datasets/AmazonScience/massive (revision `ed58ac423a2f4121720918bf5301577edce4ffd3`)
- License: CC BY 4.0
- Citation: FitzGerald et al., MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset, 2022 (Amazon)
- Changes: 100 test rows each from en-US, de-DE, es-ES, hi-IN and ja-JP; options are in English.

## multiwoz

- Source: https://huggingface.co/datasets/pfb30/multi_woz_v22 (revision `a65d2d3261c42a309b634179c186628bf35d9298`)
- License: Apache 2.0
- Citation: Zang et al., MultiWOZ 2.2: A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines, 2020
- Changes: The input is the conversation so far as a list of messages, ending with a user turn. Only user turns with at least two earlier turns and at most one active intent are used; 'none' means no active intent, such as a thank-you.

## jailbreak

- Source: https://huggingface.co/datasets/jackhhao/jailbreak-classification (revision `2f2ceeb39658696fd3f462403562b6eea5306287`)
- License: Apache 2.0
- Citation: jackhhao/jailbreak-classification on Hugging Face
- Changes: The whole test split less one repeated prompt (399 rows, 140 jailbreaks); calibration rows from the train split.

## civil_comments

- Source: https://huggingface.co/datasets/google/civil_comments (revision `f2970eb3a55777454c94069077cc8d9b5866312d`)
- License: CC0 1.0
- Citation: Borkan et al., Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification, 2019 (Jigsaw)
- Changes: Toxic means a toxicity score of 0.5 or more. Sampled half toxic and half not. Contains offensive language.

## helpsteer2

- Source: https://huggingface.co/datasets/nvidia/HelpSteer2 (revision `990b2711a36180dd19d9c94b8627844866f8982a`)
- License: CC BY 4.0
- Citation: Wang et al., HelpSteer2: Open-source dataset for training top-performing reward models, 2024 (NVIDIA)
- Changes: Human helpfulness ratings 0 to 4 mapped to five levels, sampled evenly per level. Test rows from the validation split, calibration rows from the train split. Human ratings are noisy, so mean absolute error and within-one accuracy are reported next to exact accuracy.

## wnut17

- Source: https://huggingface.co/datasets/leondz/wnut_17 (revision `1ac0b6d18c8a1a1d1606b24bef78b8af2f92d2d9`)
- License: CC BY 4.0
- Citation: Derczynski et al., Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition, 2017
- Changes: Rows with at most one entity of each type; four in five rows mention at least one entity. Tokens are joined with spaces, as in the source.

Civil Comments contains offensive language, and the jailbreak prompts try to
make assistants misbehave. Both are included because detecting them is the task.
