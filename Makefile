
define HELP
make start-all    # Start all containers
make stop-all     # Stop all containers
make start        # Start all containers but edge

make start-edge   # Start edge container

make build        # Build all containers
make setup        # Install prerequisites

make test         # Run test
  VERBOSE=1         # Show more detailed information
make clean        # Remove demo containers and images
make show-logs    # Show edge logs
endef

#----------------------------------------------------------------------------------------------

DEMO_STEM=edgerealtimevideoanalytics
SPEC=docker-compose.local.yaml
SUDO=sudo

ifeq ($(GPU),1)
DOCKER=docker
GPU_OPT=--gpus all
else
DOCKER=docker
GPU_OPT=
endif

start:
	@docker-compose -f $(SPEC) up

start-all:
	@$(DOCKER) run --rm -d --name edge --network host -it $(GPU_OPT) $(DEMO_STEM)_redisedge:latest
	@docker-compose -f $(SPEC) up -d --force-recreate

stop-all:
	@docker-compose -f $(SPEC) down
	@docker stop edge

start-edge:
	@$(DOCKER) run --rm --name edge --network host -it $(GPU_OPT) $(DEMO_STEM)_redisedge:latest

clean:
	@./tests/pplcount.sh clean

show-logs:
	@docker logs edge

build:
	@docker-compose -f docker-compose.yaml build

setup:
	@$(SUDO) ./sbin/getcompose
	@$(SUDO) ./sbin/getgitlfs
	@git lfs install; git lfs fetch; git lfs checkout

test:
	@./tests/pplcount.sh

.PHONY: start-edge start start-all stop stop-all show-logs build setup test clean help

#----------------------------------------------------------------------------------------------

ifneq ($(HELP),) 
ifneq ($(filter help,$(MAKECMDGOALS)),)
HELPFILE:=$(shell mktemp /tmp/make.help.XXXX)
endif
endif

help:
	$(file >$(HELPFILE),$(HELP))
	@echo
	@cat $(HELPFILE)
	@echo
	@-rm -f $(HELPFILE)
