import subprocess

takes = {
    "{{pc}}": ("pfPC_{}",),
    "{{mli}}": ("pfMLI_{}",),
}
procs = []
for i in range(11):
    for tag, name in takes.items():
        # setup plast_{name.format(i)} balanced.hdf5 plast_{name.format(i)}.json 30 13:00:00
        procs.append(
            subprocess.Popen(
                f"setup plast_{name.format(i)} balanced.hdf5 plast_{name.format(i)}.json"
                + " 30 13:00:00",
                shell=True
            )
        )

print("Processes started.")
[p.wait() for p in procs]
print("Processes done.")
#
# for i in range(11):
#     for j in range(11):
#         subprocess
