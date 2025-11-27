"""
Physics topic keywords for extraction
"""

PHYSICS_TOPICS = {
    # Electric Charges and Fields
    'Coulombs Law': ['coulomb', 'coulombs law', 'electrostatic force'],
    'Electric Field': ['electric field', 'field intensity', 'field lines'],
    'Gauss Theorem': ['gauss', 'gaussian surface', 'gauss theorem'],
    'Electric Dipole': ['dipole', 'dipole moment'],
    'Electric Flux': ['electric flux', 'flux'],

    # Current Electricity
    'Ohms Law': ['ohm', 'resistance', 'resistivity', 'specific resistance'],
    'Kirchhoff Laws': ['kirchhoff', 'junction rule', 'loop rule'],
    'Wheatstone Bridge': ['wheatstone', 'bridge'],
    'Potentiometer': ['potentiometer', 'e.m.f'],
    'Meter Bridge': ['meter bridge'],
    'Ammeter Voltmeter': ['ammeter', 'voltmeter', 'galvanometer'],
    'Current Drift Velocity': ['current', 'drift velocity', 'current density'],
    'Thermistor': ['thermistor'],

    # Electromagnetic
    'Biot Savart Law': ['biot', 'savart'],
    'Ampere Law': ['ampere', 'amperes law'],
    'Magnetic Field': ['magnetic field', 'magnetic intensity'],
    'Bar Magnet': ['bar magnet', 'magnetic dipole'],
    'Faraday Law': ['faraday', 'electromagnetic induction', 'induced emf'],
    'Lenz Law': ['lenz', "lenz's law"],
    'Self Inductance': ['self inductance', 'inductance', 'inductor'],
    'Mutual Inductance': ['mutual inductance'],
    'Eddy Currents': ['eddy current'],
    'AC Circuit': ['alternating current', 'a.c', 'ac circuit'],
    'Transformer': ['transformer'],
    'Choke Coil': ['choke coil'],

    # Optics
    'Reflection': ['reflection', 'mirror', 'spherical mirror'],
    'Refraction': ['refraction', 'refractive index', 'snell'],
    'Total Internal Reflection': ['total internal reflection', 'critical angle'],
    'Lens': ['lens', 'convex lens', 'concave lens', 'lens formula'],
    'Prism': ['prism', 'dispersion', 'deviation'],
    'Optical Instruments': ['microscope', 'telescope'],
    'Optical Fibre': ['optical fibre', 'fiber optics'],
    'Interference': ['interference', 'young', 'double slit'],
    'Diffraction': ['diffraction'],
    'Polarization': ['polarization', 'polaroid', 'malus'],
    'Resolving Power': ['resolving power'],
    'LASER': ['laser'],

    # Modern Physics
    'Photoelectric Effect': ['photoelectric', 'photon', 'work function'],
    'De Broglie': ['de broglie', 'matter wave'],
    'Bohr Model': ['bohr', 'atomic model', 'hydrogen spectrum'],
    'X-rays': ['x-ray', 'x ray'],
    'Radioactivity': ['radioactive', 'radioactivity', 'decay'],
    'Nuclear': ['nuclear', 'nucleus', 'binding energy'],

    # Semiconductor
    'PN Junction': ['p-n junction', 'pn junction', 'diode'],
    'Rectifier': ['rectifier', 'half wave', 'full wave'],
    'Transistor': ['transistor', 'amplifier'],
    'Logic Gates': ['logic gate', 'and gate', 'or gate', 'not gate', 'nand', 'nor'],
    'LED': ['led', 'light emitting diode'],
    'Zener Diode': ['zener', 'voltage regulator'],

    # Electromagnetic Waves
    'EM Spectrum': ['electromagnetic wave', 'em spectrum', 'electromagnetic spectrum'],
    'TV Transmission': ['television', 'antenna', 'transmission'],
}

# Question pattern keywords
DERIVATION_KEYWORDS = ['derive', 'establish', 'prove', 'deduce', 'obtain expression']
DEFINITION_KEYWORDS = ['define', 'what is', 'what are', 'explain']
NUMERICAL_KEYWORDS = ['calculate', 'find', 'determine', 'compute']
