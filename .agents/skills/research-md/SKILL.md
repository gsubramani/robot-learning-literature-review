---
name: research-md
description: Performing research and generating markdown reports
---
* Research needs to go in the .ai/research directory
* a top level index is present in the .ai/research/index.md
* each research topic should have its own directory under .ai/research/rs-<timestamp>-<topic_name>


# Structure of research topic sub folder
* Each research topic should have its own directory under .ai/research/rs-<timestamp>-<topic_name>
* each topic should have a `.ai/research/rs-<timestamp>-<topic_name>/summary.md` file. This should contain all the information along with links to further information
* further information is placed in subfolder `.ai/research/rs-<timestamp>-<topic_name>/details/<items>.md`
* subdirectory for sources under `.ai/research/rs-<timestamp>-<topic_name>/sources`
  - sources would contain the full text of all the information
  - summary of the sources
  - after the summary create the links and references
  - if the information is dense then a sub folder can be created


# How to do research
* If the user provides links or sources prioritize those. 
* Search the internet, always search the internet for all tasks. 
* Before parsing files on the internet, audit the website URL and make sure it is a legitimate website. Only after auditing the website URL are you allowed to read it directly. When in doubt, ask the user


