#!/usr/bin/env bash
# verify_reproducibility.sh: Clean-clone environment reproduction script (M5.P5).
# Verifies that repository installs and passes all tests from a fresh clone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMP_DIR="/tmp/lexsi_fresh_clone_$(date +%s)"
LOG_FILE="${REPO_DIR}/results/runs/m5_reproducibility_check.log"

mkdir -p "${REPO_DIR}/results/runs"

echo "=================================================================" | tee "${LOG_FILE}"
echo "   LEXSI TRACK 3: AUTOMATED CLEAN-CLONE REPRODUCIBILITY TEST     " | tee -a "${LOG_FILE}"
echo "=================================================================" | tee -a "${LOG_FILE}"
echo "Source Repository: ${REPO_DIR}" | tee -a "${LOG_FILE}"
echo "Temporary Clone:   ${TEMP_DIR}" | tee -a "${LOG_FILE}"
echo "Timestamp:         $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${LOG_FILE}"

cleanup() {
    if [[ -d "${TEMP_DIR}" ]]; then
        echo "Cleaning up temporary clone directory: ${TEMP_DIR}" | tee -a "${LOG_FILE}"
        rm -rf "${TEMP_DIR}"
    fi
}
trap cleanup EXIT

echo "1. Cloning repository to isolated temporary directory..." | tee -a "${LOG_FILE}"
git clone "file://${REPO_DIR}" "${TEMP_DIR}" >> "${LOG_FILE}" 2>&1

cd "${TEMP_DIR}"

echo "2. Validating commit and branch status..." | tee -a "${LOG_FILE}"
git log -1 --oneline | tee -a "${LOG_FILE}"

echo "3. Running unit test suite inside fresh clone..." | tee -a "${LOG_FILE}"
python3 -m pytest -q >> "${LOG_FILE}" 2>&1

TEST_EXIT_CODE=$?
if [[ ${TEST_EXIT_CODE} -eq 0 ]]; then
    echo "SUCCESS: All tests passed cleanly from fresh clone!" | tee -a "${LOG_FILE}"
    echo "VERDICT: REPRODUCIBILITY_VERIFIED" | tee -a "${LOG_FILE}"
else
    echo "FAILURE: Tests failed with exit code ${TEST_EXIT_CODE}" | tee -a "${LOG_FILE}"
    echo "VERDICT: REPRODUCIBILITY_FAILED" | tee -a "${LOG_FILE}"
    exit ${TEST_EXIT_CODE}
fi

echo "=================================================================" | tee -a "${LOG_FILE}"
echo "Reproducibility log saved to ${LOG_FILE}"
