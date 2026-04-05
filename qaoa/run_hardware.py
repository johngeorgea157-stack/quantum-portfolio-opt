"""
Authenticates with IBM Quantum and submits the QAOA circuit to real hardware.
WARNING: Execution is subject to real-world cloud queue times.
"""
import os
import json
import getpass
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RealSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qaoa_circuit import create_qaoa_circuit

def authenticate():
    print("Authenticating with IBM Quantum...")
    possible_paths = ["../apikey.json", "apikey.json"]
    api_key_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            api_key_path = path
            break

    if api_key_path:
        with open(api_key_path, "r") as f:
            token_data = json.load(f)
            ibm_token = token_data.get("apikey")
    else:
        ibm_token = getpass.getpass("Please paste your IBM Quantum API Token here: ")

    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=ibm_token, set_as_default=True, overwrite=True)
    return QiskitRuntimeService()

def main():
    service = authenticate()
    
    print("\nBuilding quantum circuit...")
    qc = create_qaoa_circuit()

    print("\nFinding the least busy quantum computer...")
    backend = service.least_busy(simulator=False, operational=True)
    print(f"✅ Selected Backend: {backend.name}")

    print("\nTranspiling circuit for target hardware...")
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_circuit = pass_manager.run(qc)

    print("\nSubmitting job to real hardware queue...")
    real_sampler = RealSampler(mode=backend)
    job = real_sampler.run([isa_circuit], shots=1024)
    
    print("\n🚀 Job successfully submitted!")
    print(f"Job ID: {job.job_id()}")
    print("Dashboard: https://quantum.ibm.com/jobs")
    print("\n(Note: Hardware queues may take 10-30 minutes. Check the dashboard link to monitor progress!)")

if __name__ == "__main__":
    main()
