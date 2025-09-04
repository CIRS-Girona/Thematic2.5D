# Paper

This is the JOSS abstract/paper describing the uwmm-baseline release

## Contents

Contents are:

| File      | Description |
| ---       | ---         |
| paper.md  | The primary markdown text file |
| paper.bib | The bibliography file |
| paper.pdf | The compiled PDF file |


## Compile to PDF with Docker as follows

```
docker run --rm --volume $PWD/paper:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara
```
