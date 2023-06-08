import warnings
import itertools
from tqdm import tqdm
from rdkit import Chem, rdBase

warnings.filterwarnings('ignore')
rdBase.DisableLog('rdApp.error')

# ===========================
class DataPreprocessor:
    
    def __init__(
        self, 
        input_file, 
        output_file, 
        smarts,
        attachment_point_token,
        dec_min_len=3,
        cuts=1
    ):
        """
        input_file: the original SMILES dataset
        output_file: the generated file that contains (scaffold, decorations, smiles) pairs
        cuts: the desired number of cuts (decorations) to perform
        smarts: split the bond that is not in a ring
        attachment_point_token: '*'
        """
        self.input_file = input_file
        self.output_file = output_file
        self.cuts = cuts
        self.smarts = smarts
        self.attachment_point_token = attachment_point_token
        self.dec_min_len = dec_min_len
        
    # Define the cutting function according to self.smarts
    def get_matches(self, mol):
        """
        mol: a mol object
        return: a list of (atom1, atom2) pairs that can cut the mol between atom1 and atom2
        """
        matches = set()
        matches |= set(tuple(sorted(match)) for match in mol.GetSubstructMatches(self.smarts))

        return matches

    # Given a number of cuts, enumerate all possible combinations of scaffolds and decorations
    def enumerate_scaffold_decorations(self, mol):
        """
        mol: a mol object
        return: a list of possible (scaffold, decorations) tuples
        """
        matches = self.get_matches(mol)
        combined_mols = set()
        
        # Select the number of cuts pairs from the matched combinations
        for atom_pairs_to_cut in itertools.combinations(matches, self.cuts): 
            bonds_to_cut = list(sorted(mol.GetBondBetweenAtoms(aidx, oaidx).GetIdx() for aidx, oaidx in atom_pairs_to_cut)) # Get the bond IDs between the (atom1 and atom2) pairs
            attachment_point_idxs = [(i, i) for i in range(len(bonds_to_cut))]
            cut_mol = Chem.FragmentOnBonds(mol, bondIndices = bonds_to_cut, dummyLabels = attachment_point_idxs) # Obtain the substructures of the molecule with the attachment point  token
            
            # Set the representation of all atoms in the molecule with [atom:i]
            for atom in cut_mol.GetAtoms(): 
                if atom.GetSymbol() == self.attachment_point_token:
                    num = atom.GetIsotope()
                    atom.SetIsotope(0)
                    atom.SetProp('molAtomMapNumber', str(num))        
            cut_mol.UpdatePropertyCache()
            fragments = Chem.GetMolFrags(cut_mol, asMols=True, sanitizeFrags=True)
            
            # Save the generated scaffold and decorations
            scaffold = None
            decorations = []
            # Detect whether there is a scaffold and use the fragement with the same number of cuts as scaffold
            if self.cuts == 1:
                # Calculate the length of each frag
                len_frag0 = len([atom for atom in fragments[0].GetAtoms()])
                len_frag1 = len([atom for atom in fragments[1].GetAtoms()])
                if len_frag0 >= len_frag1 and len_frag1 >= self.dec_min_len:
                    combined_mols.add(tuple([fragments[0], tuple([fragments[1]])]))
                elif len_frag0 < len_frag1 and len_frag0 >= self.dec_min_len:
                    combined_mols.add(tuple([fragments[1], tuple([fragments[0]])]))
            else:
                # Calculate the number of atoms for each frag
                len_frags = []
                for frag in fragments:
                    len_frags.append(frag.GetNumAtoms())
                    
                for frag in fragments:
                    num_attachment_points = len([atom for atom in frag.GetAtoms() if atom.GetSymbol() == self.attachment_point_token]) # Count the number of attachment point tokens of every slice
                    # Decide the scaffold and decorations
                    if num_attachment_points == self.cuts and not scaffold:
                        if frag.GetNumAtoms() == max(len_frags):
                            scaffold = frag
                    else:
                        decorations.append(frag)
                if scaffold:
                    combined_mols.add(tuple([scaffold, tuple(decorations)]))

        return list(combined_mols)

    def read_smiles_from_file(self):
        """
        return: a list of SMILES data
        """
        smiles = []
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.rstrip('\n')
                smiles.append(line)

        return smiles

    # Write the (scaffold, decorations, mol) to output_file
    def write_smiles_to_file(self, smiles):
        """
        smiles: a list of smiles data
        return: covert these SMILES data into (scaffold, decorations, smiles) pairs and save to output file
        """
        with open(self.output_file, 'w+') as fout:
            for smi in tqdm(smiles):
                mol = Chem.MolFromSmiles(smi)
                combined_mols = self.enumerate_scaffold_decorations(mol)  # Split the mol to (scaffold, decorations, smiles) pair
                
                for row in combined_mols:
                    scaffold = Chem.MolToSmiles(row[0], rootedAtAtom=0, canonical=False) #  Split the scaffold from the mol object
                    scaffold_positions = [] # Find the order of the decorations
                    scaffold_positions = [scaffold.find('[' + self.attachment_point_token +':' + str(i) +']') for i in range(self.cuts)]  
                    sorted_indexs = sorted(range(self.cuts), key = lambda k: scaffold_positions[k]) # Obtain the indexs of the order in which decorations appear in the scaffold
                    
                    decorations = []
                    for i in range(self.cuts):
                        scaffold = scaffold.replace('[' + self.attachment_point_token + ':' + str(i) + ']', self.attachment_point_token) # Replace the '[:*\d]' to '*'
                        dec_smiles = Chem.MolToSmiles(row[1][i], rootedAtAtom=0, canonical=False) # Covert the mol of decoration to SMILES
                        decorations.append(dec_smiles.replace('[' + self.attachment_point_token + ':' + str(i) + ']', '')) # Replace the '[:*\d]' to ''
                        decorations = [dec for _, dec in sorted(zip(sorted_indexs, decorations))]  # Sort the decorations according to the sorted_indexs
                    # Check if the combination is valid
                    combined_smiles = scaffold.replace(self.attachment_point_token, '{}').format(*decorations) # Replace '*' to multi-substrings of  decorations 
                    mol = Chem.MolFromSmiles(combined_smiles)
                    
                    # Check if the combination is the same 
                    if mol and Chem.MolToSmiles(mol) == Chem.MolToSmiles(Chem.MolFromSmiles(smi)):
                        if self.cuts > 1:
                            decorations = ','.join(decorations)
                        else:
                            decorations = ''.join(decorations)
                        #row_smiles = '{};{};{}'.format(scaffold, decorations, combined_smiles) 
                        row_smiles = '{};{};{}'.format(scaffold, decorations, Chem.MolToSmiles(mol))
                        fout.write('{}\n' .format(row_smiles))

                        
                        