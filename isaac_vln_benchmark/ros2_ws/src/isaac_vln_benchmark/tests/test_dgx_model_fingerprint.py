from isaac_vln_benchmark.dgx_model_fingerprint import model_files_from_command


def test_model_fingerprint_expands_all_shards_and_mmproj():
    command = (
        "/opt/llama-server -m /models/Step-IQ3-00001-of-00002.gguf "
        "--mmproj /models/mmproj-f16.gguf --port 8096"
    )
    assert model_files_from_command(command) == [
        "/models/Step-IQ3-00001-of-00002.gguf",
        "/models/Step-IQ3-00002-of-00002.gguf",
        "/models/mmproj-f16.gguf",
    ]
