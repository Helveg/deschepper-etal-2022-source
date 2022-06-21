with open("plasticity_template.json", "r") as f:
    contents = f.read()

axes = {
    "pfMLI": "{{mli}}",
    "pfPC": "{{pc}}",
}

for i in range(11):
    for axis, tag in axes.items():
        name = f"plast_{axis}_{i}"
        with open(f"configs/{name}.json", "w") as f:
            f.write(
                contents.replace("{{sim}}", name)
                .replace(tag, str(i))
                .replace("_{{mli}}", "")
                .replace("_{{pc}}", "")
            )

for i in range(11):
    for j in range(11):
        name = f"plast_pfMLI_{i}_pfPC_{j}"
        with open(f"configs/{name}.json", "w") as f:
            f.write(
                contents.replace("{{sim}}", name)
                .replace("{{mli}}", str(i))
                .replace("{{pc}}", str(j))
            )
