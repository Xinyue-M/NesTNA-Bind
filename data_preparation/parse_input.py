def parse_input(input_file):
    
    input_info = []
    with open(input_file, 'r') as f:
        lines = f.read().split("\n")
        for line in lines[:-1]:
            pdbid, chainid, rnaids = line.split("\t")[0].split(':')
            binding_info = line.split("\t")[1].split(':')
            input_info.append((pdbid, chainid, rnaids, binding_info))
            # print(pdbid, chainid, rnaids, binding_info)
    return input_info

def parse_dmasif(input_file):
    input_info = []
    with open(input_file, 'r') as f:
        lines = f.read().split("\n")
        for line in lines:
            info = line.split("_")
            pdbid = info[0]
            p0 = info[1]
            p1 = info[2]
            if len(p0) == 1 and len(p1) == 1:
                input_info.append((pdbid, p0, p1, None))
    return input_info
