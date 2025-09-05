# Paper

This is the JOSS abstract/paper describing the uwmm-baseline release


## Compile to PDF with Docker as follows

```
docker run --rm --volume $PWD:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara
```

**Note**: Make sure you are in the JOSS folder before running the above command. The `paper.md` and `paper.bib` must be present along with any media files referenced in order for the compilation to succeed.
