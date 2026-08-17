UV_CACHE_DIR ?= /tmp/pvzwm-uv-cache
export UV_CACHE_DIR

.PHONY: macos-deps setup setup-intel play play-teacher import-teacher-run create-adventure-save inspect-teacher verify-teacher verify-teacher-resume verify-teacher-campaign reference-baseline save-slots fork-save fork-checkpoint test-goty test-renderers test-campaign test-playthroughs test-modes test-mode-playthroughs test-saves test-string-format test-audio

macos-deps:
	@test "$$(uname -s)" = "Darwin" || { echo "macos-deps is only for macOS." >&2; exit 2; }
	@command -v brew >/dev/null || { echo "Install Homebrew from https://brew.sh first." >&2; exit 2; }
	brew install cmake ninja sdl2 jpeg-turbo libpng libogg libvorbis libopenmpt mpg123

setup:
	scripts/setup_portable_backend.sh

setup-intel:
	@test "$$(uname -s)" = "Darwin" || { echo "setup-intel is only for macOS." >&2; exit 2; }
	PVZ_MACOS_ARCHS=x86_64 \
	PVZ_PORTABLE_BUILD_DIR=vendor/PvZ-Portable/build-macos-x86_64 \
	scripts/setup_portable_backend.sh

play:
	PVZ_GOTY_SAVE_SLOT="$(SAVE_SLOT)" scripts/play_goty.sh

play-teacher:
	@test -n "$(SAVE_SLOT)" || { echo "Usage: make play-teacher SAVE_SLOT=world1-clean" >&2; exit 2; }
	PVZ_GOTY_TEACHER=1 PVZ_GOTY_SAVE_SLOT="$(SAVE_SLOT)" scripts/play_goty.sh

import-teacher-run:
	@test -n "$(RUN)" || { echo "Usage: make import-teacher-run RUN='/path/to/legacy-run' SAVE_SLOT=world1-clean" >&2; exit 2; }
	@test -n "$(SAVE_SLOT)" || { echo "Usage: make import-teacher-run RUN='/path/to/legacy-run' SAVE_SLOT=world1-clean" >&2; exit 2; }
	uv run scripts/import_teacher_run.py "$(RUN)" --save-slot "$(SAVE_SLOT)" \
		$(if $(CAMPAIGN),--campaign "$(CAMPAIGN)",)

create-adventure-save:
	@test -n "$(SAVE_SLOT)" || { echo "Usage: make create-adventure-save SAVE_SLOT=before-5-10 LEVEL=5-10" >&2; exit 2; }
	@test -n "$(LEVEL)" || { echo "Usage: make create-adventure-save SAVE_SLOT=before-5-10 LEVEL=5-10" >&2; exit 2; }
	@test "$$(uname -s)" = "Darwin" || { echo "create-adventure-save is only for macOS Metal builds." >&2; exit 2; }
	PVZWM_BUILD_METAL_BRIDGE=ON scripts/setup_portable_backend.sh
	uv run scripts/create_goty_adventure_save.py \
		--binary vendor/PvZ-Portable/build-macos/pvzwm-metal-bridge \
		--resources data/pvz-goty --save-slot "$(SAVE_SLOT)" \
		--target-level "$(LEVEL)" $(if $(SEED),--seed "$(SEED)") \
		$(if $(DIRECT),--direct-target)

inspect-teacher:
	@test -n "$(RUN)" || { echo "Usage: make inspect-teacher RUN='/path/to/teacher-run'" >&2; exit 2; }
	uv run scripts/inspect_teacher_run.py "$(RUN)" $(if $(EXPECTED),--expected-checkpoints "$(EXPECTED)",)

verify-teacher: inspect-teacher
	uv run scripts/verify_teacher_replay.py "$(RUN)" \
		--binary vendor/PvZ-Portable/build-macos/pvz-portable \
		--resources data/pvz-goty $(if $(BLACKOUT),--blackout) $(if $(HIDDEN),--hidden)

verify-teacher-resume: inspect-teacher
	PVZWM_BUILD_METAL_BRIDGE=ON scripts/setup_portable_backend.sh
	uv run scripts/verify_teacher_resume.py "$(RUN)" \
		--binary vendor/PvZ-Portable/build-macos/pvzwm-metal-bridge \
		--resources data/pvz-goty \
		$(if $(EXPECTED),--expected-checkpoints "$(EXPECTED)",)

verify-teacher-campaign:
	@test -n "$(RUN)" || { echo "Usage: make verify-teacher-campaign RUN='/path/to/campaign'" >&2; exit 2; }
	@test "$$(uname -s)" = "Darwin" || { echo "verify-teacher-campaign is only for macOS Metal builds." >&2; exit 2; }
	uv run scripts/inspect_teacher_run.py "$(RUN)" \
		--expected-checkpoints 50 --require-complete-adventure
	PVZWM_BUILD_METAL_BRIDGE=ON scripts/setup_portable_backend.sh
	uv run scripts/verify_teacher_campaign_replay.py "$(RUN)" \
		--binary vendor/PvZ-Portable/build-macos/pvz-portable \
		--resources data/pvz-goty --timeout "$(or $(TIMEOUT),3600)" \
		$(if $(BLACKOUT),--blackout) $(if $(HIDDEN),--hidden)
	uv run scripts/verify_teacher_resume.py "$(RUN)" \
		--binary vendor/PvZ-Portable/build-macos/pvzwm-metal-bridge \
		--resources data/pvz-goty --expected-checkpoints 50

reference-baseline:
	@test -n "$(REFERENCE_ROOT)" || { echo "Usage: make reference-baseline REFERENCE_ROOT='/path/to/PvZ' REFERENCE_VERSION=1.2.0.1073" >&2; exit 2; }
	@test -n "$(REFERENCE_VERSION)" || { echo "REFERENCE_VERSION is required." >&2; exit 2; }
	uv run scripts/manifest_goty_reference.py "$(REFERENCE_ROOT)" \
		--version "$(REFERENCE_VERSION)" \
		--distribution "$(or $(REFERENCE_DISTRIBUTION),steam-windows)" \
		--output data/reference/local/goty-reference.json \
		--acceptance-output data/reference/local/goty-acceptance.json \
		--require-executable

save-slots:
	scripts/play_goty.sh --list-save-slots

fork-save:
	@test -n "$(FROM)" || { echo "Usage: make fork-save FROM=source TO=destination" >&2; exit 2; }
	@test -n "$(TO)" || { echo "Usage: make fork-save FROM=source TO=destination" >&2; exit 2; }
	scripts/fork_goty_save.sh "$(FROM)" "$(TO)"

fork-checkpoint:
	@test -n "$(CHECKPOINT)" || { echo "Usage: make fork-checkpoint CHECKPOINT='/path/to/checkpoint' TO=slot" >&2; exit 2; }
	@test -n "$(TO)" || { echo "Usage: make fork-checkpoint CHECKPOINT='/path/to/checkpoint' TO=slot" >&2; exit 2; }
	scripts/fork_teacher_checkpoint.sh "$(CHECKPOINT)" "$(TO)"

test-goty:
	uv run scripts/smoke_portable.py \
		--binary vendor/PvZ-Portable/build-macos/pvzwm-bridge \
		--resources data/pvz-goty --no-xvfb

test-renderers:
	@test "$$(uname -s)" = "Darwin" || { echo "test-renderers is only for macOS." >&2; exit 2; }
	PVZWM_BUILD_METAL_BRIDGE=ON scripts/setup_portable_backend.sh
	uv run scripts/compare_goty_renderers.py \
		--opengl-binary vendor/PvZ-Portable/build-macos/pvzwm-bridge \
		--metal-binary vendor/PvZ-Portable/build-macos/pvzwm-metal-bridge \
		--resources data/pvz-goty \
		--max-mean-absolute-error 0.05 \
		--max-different-pixel-fraction 0.025

test-campaign:
	@test "$$(uname -s)" = "Darwin" || { echo "test-campaign is only for macOS." >&2; exit 2; }
	PVZWM_BUILD_METAL_BRIDGE=ON scripts/setup_portable_backend.sh
	uv run scripts/audit_goty_campaign.py \
		--binary vendor/PvZ-Portable/build-macos/pvzwm-bridge \
		--candidate-binary vendor/PvZ-Portable/build-macos/pvzwm-metal-bridge \
		--resources data/pvz-goty --levels 1..50 \
		--exercise-special-actions

test-playthroughs:
	@test "$$(uname -s)" = "Darwin" || { echo "test-playthroughs is only for macOS." >&2; exit 2; }
	PVZWM_BUILD_METAL_BRIDGE=ON scripts/setup_portable_backend.sh
	uv run scripts/audit_goty_playthroughs.py \
		--binary vendor/PvZ-Portable/build-macos/pvzwm-bridge \
		--candidate-binary vendor/PvZ-Portable/build-macos/pvzwm-metal-bridge \
		--resources data/pvz-goty --levels 1..50 \
		--max-ticks 360000

test-modes:
	@test "$$(uname -s)" = "Darwin" || { echo "test-modes is only for macOS." >&2; exit 2; }
	PVZWM_BUILD_METAL_BRIDGE=ON scripts/setup_portable_backend.sh
	uv run scripts/audit_goty_modes.py \
		--binary vendor/PvZ-Portable/build-macos/pvzwm-bridge \
		--candidate-binary vendor/PvZ-Portable/build-macos/pvzwm-metal-bridge \
		--resources data/pvz-goty --modes 1..70

test-mode-playthroughs:
	@test "$$(uname -s)" = "Darwin" || { echo "test-mode-playthroughs is only for macOS." >&2; exit 2; }
	PVZWM_BUILD_METAL_BRIDGE=ON scripts/setup_portable_backend.sh
	uv run scripts/audit_goty_mode_playthroughs.py \
		--binary vendor/PvZ-Portable/build-macos/pvzwm-bridge \
		--candidate-binary vendor/PvZ-Portable/build-macos/pvzwm-metal-bridge \
		--resources data/pvz-goty --modes 1,2,3,4,22,30,31,37,39,40,45,51,52,53,54,55,56,57,58,59,61,62,63,64,65,66,67,68,69 \
		--max-ticks 150000

test-saves:
	@test "$$(uname -s)" = "Darwin" || { echo "test-saves is only for macOS." >&2; exit 2; }
	PVZWM_BUILD_METAL_BRIDGE=ON scripts/setup_portable_backend.sh
	uv run scripts/audit_goty_saves.py \
		--binary vendor/PvZ-Portable/build-macos/pvzwm-bridge \
		--candidate-binary vendor/PvZ-Portable/build-macos/pvzwm-metal-bridge \
		--resources data/pvz-goty

test-string-format:
	PVZWM_BUILD_TESTS=ON scripts/setup_portable_backend.sh
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		build_dir=vendor/PvZ-Portable/build-macos; \
	else \
		build_dir=vendor/PvZ-Portable/build; \
	fi; \
	"$$build_dir/pvzwm-string-format-test"

test-audio:
	PVZWM_BUILD_TESTS=ON scripts/setup_portable_backend.sh
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		build_dir=vendor/PvZ-Portable/build-macos; \
	else \
		build_dir=vendor/PvZ-Portable/build; \
	fi; \
	"$$build_dir/pvzwm-openmpt-tempo-test"
