.PHONY: all build debug clean profile bench cuobjdump rsession rclone rsetup rdl

CMAKE := cmake

BUILD_DIR := build
BENCHMARK_DIR := benchmark_results

SESSION := sgemm
# keep ssh connection alive for the fast reconnects
SSH := ssh -o ProxyCommand='colab ssh --proxy-mode -s $(SESSION)' \
	   -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
	   -o ControlMaster=auto -o ControlPath=/tmp/ssh-colab-$(SESSION) -o ControlPersist=10m \
	   root@colab
GPU := T4
RDIR := /content/sgemm
REPO_URL := https://github.com/dmalt/SGEMM_CUDA.git 

all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && \
		$(CMAKE) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-lineinfo" ..
	@$(MAKE) -C $(BUILD_DIR)

debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug ..
	@$(MAKE) -C $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)


# Usage: make profile KERNEL=<integer> PREFIX=<optional string>
profile: build
	@ncu --set full \
		--export $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL) \
		--force-overwrite $(BUILD_DIR)/sgemm \
		$(KERNEL)

bench: build
	@bash gen_benchmark_results.sh

# REMOTE COMMANDS
rsession:
	@colab sessions | grep -q '$(SESSION)' || colab new -s $(SESSION) --gpu $(GPU)
	@$(SSH) "grep -q cuda/bin ~/.bashrc || sed -i '1i export PATH=/usr/local/cuda/bin:\$$PATH' ~/.bashrc"
rclone:
	@$(SSH) "rm -rf $(RDIR) && git clone $(REPO_URL) $(RDIR)"
rsetup: rsession rclone
rprofile:
	@$(SSH) 'cd $(RDIR) && make profile KERNEL=$(KERNEL) PREFIX=$(PREFIX)'
	@colab download -s $(SESSION) \
		"$(RDIR)/benchmark_results/$(PREFIX)kernel_$(KERNEL).ncu-rep" \
		"benchmark_results/$(PREFIX)kernel_$(KERNEL).ncu-rep"

