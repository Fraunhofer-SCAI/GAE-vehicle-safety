# ToDo prio

## Docker
- [ ] wait for andre or course input

## Spinx
 -  recover the old process, [link](https://github.com/antahiap/sphinx-test)

## Ontology
- [x] add it to auto documantation pipline

Install package 
```
 git clone https://github.com/RDFLib/pyLODE.git
 cd pyLODE
 pip install -r requirements.txt
 pyzhon steup.py
 cd pylode 
 python cli.py PATH_TO_OWL_FILE
 ```

## CI
- write the actios in, `.github\workflows\` inititated drom sphinx dcoumentation manual, [link](https://github.com/antahiap/sphinx-test)
- currently install requirements > build sphinx doc 
- [x] debug creating ontology doc
- [ ] deploy django on github pages


## Code transfer

### Document
 - [ ] write model functions

### API 
 - [ ] nrg_fts > visualize fingerprint, histogram and graph
 - [ ] run ld_data
   - [ ] fix security issue to read



# Documentation
- [x] publish report, [guide](https://coderefinery.github.io/documentation/gh_workflow/)
    - [x] add pageds to github repo, [guide](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)
    - [x] remove sphinx guide from GAE repo
    - [x] add ontology report page to github

- [ ] finish the code comments
- [ ] deploy django page template for plotly
- [ ] good sample



## updated
- docker build
- howto on making sphinx is lost
- issue with django-bootstrap5 and django-neomodel



### Docker Container

- build an image `buildah build -t gae:v0-djang`
- run the container `podman run -dp 127.0.0.1:3000:3000 gae:v0-django`
- check if it is running, `podman ps -a `
- change the tag to match dockerhub `podman tag gae:v0-django antahia/gae:v0-django`

- run a docker, `podman run gae:v0-django`
- view existing containers runing, `podman ps -a`

#### To push
- login to the docker hub, `docker login`
- commit a local docekr to a host image `docker commit a2607f8e4a16 docker.io/antahia/gae:v0-django`
- push a docker


