import subprocess
import sys

args = sys.argv

takes = {
    "{{pc}}": "pfPC_{}",
    "{{mli}}": "pfMLI_{}",
}
procs = []

if "sep" in args:
    for i in range(11):
        for tag, name in takes.items():
            # setup plast_{name.format(i)} balanced.hdf5 plast_{name.format(i)}.json 30 13:00:00
            procs.append(
                subprocess.Popen(
                    "/store/hbp/ich027/devops/setup_simulation"
                    + f" plast_{name.format(i)} balanced.hdf5 plast_{name.format(i)}.json"
                    + " 30 13:00:00",
                    shell=True
                )
            )

if "nocross" not in args:
    for i in range(11):
        for j in range(11):
                procs.append(
                    subprocess.Popen(
                        "/store/hbp/ich027/devops/setup_simulation"
                        + f" plast_pfMLI_{i}_pfPC_{j} balanced.hdf5 plast_pfMLI_{i}_pfPC_{j}.json"
                        + " 30 13:00:00",
                        shell=True
                    )
                )

print(len(procs), "processes started.")
[p.wait() for p in procs]
print("Processes done.")
#
# for i in range(11):
#     for j in range(11):
#         subprocess
