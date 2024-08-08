import _pickle as cPickle
import esm
import torch 

def init_esm():
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local("./esm2_t33_650M_UR50D.pt")
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    return esm_model, batch_converter

if __name__ == "__main__":
    
    print("start")

    # fn = "DNA-179_Test_seqs"
    # with open(fn, 'rb') as f:
    #     seq_dict = cPickle.load(f)
    # f.close()

    # esm_model, batch_converter = init_esm()
    # print(esm_model)

    # for k, v in seq_dict.items():
    #     batch_labels, batch_strs, batch_tokens = batch_converter([(k, v[0])])
    #     with torch.no_grad():
    #         results = esm_model(batch_tokens, repr_layers=[9, 19, 33], return_contacts=True)

    #     token_representations_33 = results["representations"][33][:, 1:-1, :]
    #     token_representations_19 = results["representations"][19][:, 1:-1, :]
    #     token_representations_9 = results["representations"][9][:, 1:-1, :]

    #     seq_dict[k] += [token_representations_9, token_representations_19, token_representations_33]
    #     # seq_dict[k].append(token_representations_9, token_representations_19, token_representations_33)
    #     print(f"{k} esm information added.")
    #     # break

    # with open(f"{fn}_esm", 'wb') as f:
    #     cPickle.dump(seq_dict, f)
    # f.close()

    # ============================================get esm from fasta============================================
    fn = "DNA-179_Test.fasta"
    with open(fn, 'r') as f:
        lines = f.read().split('\n')
    f.close()
    print(len(lines))

    esm_model, batch_converter = init_esm()
    seq_dict = {}

    for i in range(int(len(lines) / 2)):
        pro_id = lines[i * 2][1:]
        seq = lines[i * 2 + 1]
        batch_labels, batch_strs, batch_tokens = batch_converter([(pro_id, seq)])
        
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[9, 19, 33], return_contacts=True)

        token_representations_33 = results["representations"][33][:, 1:-1, :]
        token_representations_19 = results["representations"][19][:, 1:-1, :]
        token_representations_9 = results["representations"][9][:, 1:-1, :]
        seq_dict[pro_id] = [seq, token_representations_9, token_representations_19, token_representations_33]
        print(f"{pro_id} esm information added.")
    
    with open(f"{fn.split('.')[0]}_esm", 'wb') as f:
        cPickle.dump(seq_dict, f)
    f.close()
