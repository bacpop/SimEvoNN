import subprocess
import os


def call_subprocess(command: str, params: list, outfile=None, chdir=None):
    # When we want to pipe the result to a text file, then we have to use the outfile option.
    # If the program asks you to specify the output with -o etc. then leave the outfile param None
    if outfile:
        stdout_buffer = open(outfile, "wb", buffering=0)
    else:
        stdout_buffer = subprocess.PIPE

    popen_args = dict(
        args=[command] + params,
        preexec_fn=os.setsid,
        stdin=subprocess.DEVNULL,
        stdout=stdout_buffer,
        stderr=subprocess.PIPE,
        bufsize=0,
        cwd=chdir,
    )
    process = subprocess.Popen(**popen_args)
    stdout, stderr = process.communicate()

    return_code = process.returncode

    if return_code != 0:
        full_command = " ".join(popen_args['args'])
        raise Exception(full_command, stdout, stderr)
    retstdout = stdout.decode() if stdout is not None else None
    return return_code, retstdout


def run_maple(input_fasta_path):
    import MAPLE
    import tempfile
    import os
    input_dir= os.path.dirname(input_fasta_path)
    input_fasta= os.path.basename(input_fasta_path)
    ### First create a Maple VCF format
    temp_maple_file = tempfile.mktemp(suffix=".txt", prefix="WF_sim", dir=input_dir)
    call_subprocess("pypy", [MAPLE.createMapleFile, "--path", f"{input_dir}/", "--fasta", input_fasta, "--output", os.path.basename(temp_maple_file), "--overwrite"])
    ### Then construct a tree
    call_subprocess("pypy", [MAPLE.run_Maple, "--input", temp_maple_file, "--output", f"{input_dir}/", "--overwrite"])

    tree_path = os.path.join(input_dir, "_tree.tree")
    return tree_path, temp_maple_file


def create_maple_tree(maple_file):
    import MAPLE
    import os
    output_dir = os.path.dirname(maple_file)
    try:
        call_subprocess("pypy", [MAPLE.run_Maple, "--input", maple_file, "--output", f"{output_dir}/", "--overwrite"])
    except FileNotFoundError:
        call_subprocess("python", [MAPLE.run_Maple, "--input", maple_file, "--output", f"{output_dir}/", "--overwrite"])

    tree_path = os.path.join(output_dir, "_tree.tree")
    return tree_path
