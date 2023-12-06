---
language:
- en
license: cc-by-nc-4.0
license_name: Attribution-NonCommercial 4.0 International
tags:
- event-summarization
- background-summarization
annotations_creators:
- expert-generated
language_creators:
- expert-generated
language_details:
- en-US
pretty_name: Background Summarization
size_categories:
- 1K<n<10K
source_datasets:
- Timeline17
- Crisis
- SocialTimeline
task_categories:
- summarization
---

# Dataset Card for Background Summarization of Event Timelines

This dataset provides background text summaries for news events timelines.

## Dataset Details

### Dataset Description

Generating concise summaries of news events is a challenging natural language processing task. While journalists often curate timelines to highlight key sub-events, newcomers to a news event face challenges in catching up on its historical context. This dataset addresses this need by introducing the task of background news summarization, which complements each timeline update with a background summary of relevant preceding events. This dataset includes human-annotated backgrounds for 14 major news events from 2005--2014.

- **Curated by:** Adithya Pratapa, Kevin Small, Markus Dreyer
- **Language(s) (NLP):** English
- **License:** CC-BY-NC-4.0

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://github.com/amazon-science/background-summaries
- **Paper:** https://arxiv.org/abs/2310.16197

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

This dataset can be used for training text summarization systems. The trained systems would be capable of generating background (historical context) to a news update. To generate the background, the system takes past news updates as input.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

Systems trained on this dataset might not perform as expected on domains other than newswire. To avoid factual errors, system-generated summaries should be verified by experts before deploying in real-world.

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

### Dataset Fields

| Field | Name | Description |
| :--- | :--- | :--- |
| src | Source | Concatenated string of all the previous updates. Each update text includes the publication date. |
| z | Guidance | Update text for the current timestep. |
| tgt | Target | Background text for the current timestep. |

### Data Splits

An overview of the major events and their splits in this dataset. The last column provides the statistics for background annotations provided in this dataset.

| Split | Major event |  Sources (# timelines) |  Time period |  # updates |  len(updates) |  len(background) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Train | Swine flu |  T17 (3) |  2009 |  21 |  52 |  45 |
| Train | Financial crisis |  T17 (1) |  2008 |  65 |  115 |  147 |
| Train | Iraq war |  T17 (1) |  2005 |  155 |  41 |  162 |
| Validation | Haitian earthquake |  T17 (1) |  2010 |  11 |  100 |  61 |
| Validation | Michael Jackson death |  T17 (1) |  2009--2011 |  37 |  36 |  164 |
| Validation | BP oil spill |  T17 (5) |  2010--2012 |  118 |  56 |  219 |
| Test | NSA leak |  SocialTimeline (1) |  2014 |  29 |  45 |  50 |
| Test | Gaza conflict |  SocialTimeline (1) |  2014 |  38 |  183 |  263 |
| Test | MH370 flight disappearance |  SocialTimeline (1) |  2014 |  39 |  39 |  127 |
| Test | Yemen crisis |  Crisis (6) |  2011--2012 |  81 |  30 |  125 |
| Test | Russian-Ukraine conflict |  SocialTimeline (3) |  2014 |  86 |  112 |  236 |
| Test | Libyan crisis |  T17 (2); Crisis (7) |  2011 |  118 |  38 |  177 |
| Test | Egyptian crisis |  T17 (1); Crisis (4) |  2011--2013 |  129 |  34 |  187 |
| Test | Syrian crisis |  T17 (4); Crisis (5) |  2011--2013 |  164 |  30 |  162 |

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

Readers often find it difficult to keep track of complex news events. A background summary that provides sufficient historical context can help improve the reader's understanding of a news update. This dataset provides human-annotated backgrounds for development and evaluation of background summarization systems.

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

This dataset is built upon three popular news timeline summarization datasets, Timeline17 ([Binh Tran et al., 2013](https://dl.acm.org/doi/10.1145/2487788.2487829)), Crisis ([Tran et al., 2015](https://link.springer.com/chapter/10.1007/978-3-319-16354-3_26)), and Social Timeline ([Wang et al., 2015](https://aclanthology.org/N15-1112/)).

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

__Timeline17:__ compiled from an ensemble of news websites, this dataset provides 17 timelines spanning 9 major events from 2005--2013.

__Crisis:__ a follow-up to the Timeline17 dataset, this covers 25 timelines spanning 4 major events. While it mostly covers a subset of events from Timeline17, it adds a new event (the Yemen crisis).

__Social Timeline:__ compiled 6 timelines covering 4 major events from 2014. The timelines were collected from Wikipedia, NYTimes, and BBC.

### Annotations

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

Timelines were originally collected from various news websites (CNN, BBC, NYTimes, etc.), many events have more than one timeline. Since each timeline covers the same underlying event, we merge them using timestamps to create a single timeline per event. During this merging process, we often end up with more than one update text per timestamp with possibly duplicate content. We ask the annotators to first rewrite the input updates to remove any duplicate content. Our annotation process for each news event contains the following three steps:

1. Read the input timeline to get a high-level understanding of the event.
2. For each timestep, read the provided 'rough' update summary. Rewrite the update into a short paragraph, removing any duplicate or previously reported subevents.
3. Go through the timeline in a sequential manner and write a background summary for each timestep.

#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

We hired three professional annotators. For each timeline, we collect three independent (rewritten) update and (new) background pairs.

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

To the best of our knowledge, there is no personal or sensitive information in this dataset.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

### Limitations

__Personalized Backgrounds:__  While a background summary can be useful to any news reader, the utility can vary depending on the reader's familiarity with the event. This dataset doesn't include any backgrounds customized to individual readers.

__Local Events:__ This dataset is limited to globally popular events involving disasters and conflicts. We leave the task of collecting background summaries for local events to future work.

__Background from News Articles:__ Background summaries can also be generated directly from news articles. In this dataset, we only consider background summaries based on past news updates. We leave the extension to news articles to future work.

## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

__BibTeX:__

```bibtex
@article{pratapa-etal-2023-background,
title = {Background Summarization of Event Timelines},
author = {Pratapa, Adithya and Small, Kevin and Dreyer, Markus},
publisher = {EMNLP},
year = {2023},
url = {https://arxiv.org/abs/2310.16197},
}
```

## Glossary

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

__Major event:__ the key news story for which we are constructing a timeline. For instance, 'Egyptian Crisis', 'BP oil spill', 'MH 370 disappearance' are some of the super events from our dataset.

__Timeline:__ a series of timesteps. Each timestep in a timeline is associated with an update and a background summary. 

__Timestep:__ day of the event (`yyyy-mm-dd`).

__Update:__ a short text summary of _what's new_ in the news story. This text summarizes the latest events, specifically ones that are important to the overall story. 

__Background:__ a short text summary that provides _sufficient historical context_ for the current update. Background aims to provide the reader a quick history of the news story, without them having to read all the previous updates. Background should cover past events that help in understanding the current events described in the update.

## Dataset Card Authors

Adithya Pratapa, Kevin Small, Markus Dreyer

## Dataset Card Contact

[Adithya Pratapa](https://apratapa.xyz)
