from pypop import *
from pypop.pypop import Occupant, Site, Lattice, Reaction

import numpy as np
from pytest import raises
from scipy.stats import binomtest

def test_species():
    # Test species initialization and naming
    species = Species("A")
    assert(str(species) == "A")

    # Test occupant creation and correct species bookkeeping
    site = Site("site")
    occupant = Occupant(species, site)
    assert(species.members[0] == occupant)
    assert(len(species) == 1)

    # Test occupant removal
    species.remove(occupant)
    assert(len(species.members) == 0)
    assert(len(species) == 0)

    # Test species occupant addition
    species.add(occupant)
    assert(species.members[0] == occupant)
    assert(len(species) == 1)

    # Test second occupant addition and first removal (list bookkeeping)
    second_occupant = Occupant(species, site)
    assert(len(species) == 2)
    species.remove(occupant)
    assert(species.members[0] == second_occupant)

def test_site():
    # Test site initialization and naming
    site = Site("site")
    assert(str(site) == "site")

    # Random number generator to be used below
    rng = np.random.default_rng()

    # Test that we get None if we request a random occupant of a site that has no occupants of that species
    species = Species("A")
    assert(site.get_random_occupant(species, rng) is None)
    
    # Test occupant creation and on-site bookkeeping
    first_occupant = Occupant(species, site)
    second_occupant = Occupant(species, site)
    assert(site.species_abundance(species) == 2)

    # Test that we get a random occupant from the choice on the site
    assert(site.get_random_occupant(species, rng) in (first_occupant, second_occupant))

    # Test removal of occupant from site
    site.remove(first_occupant)
    assert(site.species_abundance(species) == 1)

def test_occupant():
    # Create sites and species to be used below
    site1 = Site("site1")
    site2 = Site("site2")
    species = Species("A")
    
    # Test occupant initialization, naming and bookkeeping
    occupant = Occupant(species, site1)
    assert(hash(occupant) == occupant.id)
    assert(str(occupant) == f"{species.name}{occupant.id}")
    assert(species.members[0] == occupant)

    # Test that the occupant was assigned to the correct species and site
    assert(occupant.species == species)
    assert(occupant.site == site1)

    # Test that the abundances are correct at the different sites
    assert(site1.species_abundance(species) == 1)
    assert(site2.species_abundance(species) == 0)
    assert(site1.species_occupants[species][0] == occupant)

    # Change occupant site and test that the abundances are again correct
    occupant.set_site(site2)
    assert(site1.species_abundance(species) == 0)
    assert(site2.species_abundance(species) == 1)
    assert(site2.species_occupants[species][0] == occupant)

    # Destroy occupant and test that the abundances are again correct
    occupant.destroy()
    assert(site1.species_abundance(species) == 0)
    assert(site2.species_abundance(species) == 0)

def test_lattice():
    # Test lattice initialization
    size = (5, 5)
    lattice = Lattice(size)
    assert(lattice.size == size)
    assert(lattice.nr_sites == size[0]*size[1])
    assert(len(lattice.sites) == size[0]*size[1])

    # Test that sites were named correctly and that the sites are assigned the correct neighbors
    for i in range(size[0]):
        for j in range(size[1]):
            site = lattice.sites[i + j*size[0]]
            assert(site.id == f"{j}x{i}")
            assert(len(set(site.neighbors)) == 4)

            if i > 0:
                assert(lattice.sites[i - 1 + j*size[0]] in site.neighbors)
            else:
                assert(lattice.sites[(j+1)*size[0] - 1] in site.neighbors)
            
            if i < size[0]-1:
                assert(lattice.sites[i + 1 + j*size[0]] in site.neighbors)
            else:
                assert(lattice.sites[j*size[0]] in site.neighbors)
            
            if j > 0:
                assert(lattice.sites[i + (j-1)*size[0]] in site.neighbors)
            else:
                assert(lattice.sites[i + (size[1]-1)*size[0]] in site.neighbors)
            
            if j < size[1]-1:
                assert(lattice.sites[i + (j+1)*size[0]] in site.neighbors)
            else:
                assert(lattice.sites[i] in site.neighbors)

def test_reactions():
    # Create species, site and occupant to be used below
    species = Species("A")
    site = Site("A")
    occupant = Occupant(species, site)

    # Test parameters
    rate = 0.6
    trials = 100000
    rng = np.random.default_rng()

    # Test that we cannot create a Reaction object (needs to be subclassed)
    reaction = Reaction(rate)
    with raises(NotImplementedError) as excinfo:
        reaction(occupant, rng)

    # Test BirthReaction initialization
    birth_reaction = BirthReaction(species, rate)
    assert(birth_reaction.species == species)
    assert(birth_reaction.rate == rate)
    # Test that the reaction happens at the correct rate
    # Note, this can fail randomly, but should be fine mostly!! It is a statistical test after all.
    successes = 0
    for _ in range(trials):
        # Birth reactions are not allowed to kill the initial occupant
        assert(birth_reaction(occupant, rng) == False)
        if len(species) == 2:
            successes += 1
            species.members[1].destroy()
    assert(binomtest(successes, trials, rate).pvalue > 0.01)

    # Test DeathReaction initialization
    death_reaction = DeathReaction(species, rate)
    assert(death_reaction.species == species)
    assert(death_reaction.rate == rate)
    # Test that the reaction happens at the correct rate
    # Note, this can fail randomly, but should be fine mostly!! It is a statistical test after all.
    successes = 0
    for _ in range(trials):
        if death_reaction(occupant, rng):
            successes += 1
            occupant = Occupant(species, site)
    assert(binomtest(successes, trials, rate).pvalue > 0.01)

    # Test PredationReaction initialization
    prey_species = Species("B")
    predation_reaction = PredationReaction(species, prey_species, rate)
    assert(predation_reaction.speciesA == species)
    assert(predation_reaction.speciesB == prey_species)
    assert(predation_reaction.rate == rate)
    # Test that the reaction happens at the correct rate
    # Note, this can fail randomly, but should be fine mostly!! It is a statistical test after all.
    for _ in range(trials):
        Occupant(prey_species, site)
    assert(predation_reaction(occupant, rng) == False)
    successes = trials - len(prey_species)
    assert(binomtest(successes, trials, rate).pvalue > 0.01)

    # Test PredationBirthReaction initialization
    prey_species = Species("C")
    predationbirth_reaction = PredationBirthReaction(species, prey_species, rate)
    assert(predationbirth_reaction.speciesA == species)
    assert(predationbirth_reaction.speciesB == prey_species)
    assert(predationbirth_reaction.rate == rate)
    # Test that the reaction happens at the correct rate
    # Note, this can fail randomly, but should be fine mostly!! It is a statistical test after all.
    for _ in range(trials):
        Occupant(prey_species, site)
    assert(predationbirth_reaction(occupant, rng) == False)
    successes = trials - len(prey_species)
    assert(len(species) == successes + 1)
    assert(binomtest(successes, trials, rate).pvalue > 0.01)

def test_hop():
    # Create species, lattice and occupant for use below
    species = Species("A")
    lattice = Lattice((2, 2))
    initial_site = lattice.sites[0]
    rng = np.random.default_rng()
    occupant = Occupant(species, initial_site)
    
    # Test hop reaction initialization
    hop = Hop(species, 1.0)
    # Test that the occupant actually hops to a neighboring site
    hop(occupant, rng)
    assert(occupant.site in initial_site.neighbors)

def test_world():
    # Test parameters
    size = (2, 2)
    A = Species("A")
    B = Species("B")
 
    reactions = {
        A: [
            BirthReaction(A, 0.1),
        ],
        B: [
            DeathReaction(A, 0.1),
        ]
    }

    # Test world initialization
    world = World(
        size=size,
        initial_densities={A: 10.0, B: 10.0},
        hops={A: Hop(A, 1.0), B: Hop(A, 1.0)},
        reactions=reactions,
    )
    # We already test the lattice above

    # Run 10 steps
    world.run(10)
    # Test that the abundances are returned in the correct format
    abundances = world.abundances
    assert(set(abundances.keys()) == {"A", "B"})
    # Test that the arrays are returned in the correct format and shape
    arrays = world.asarrays()
    assert(set(arrays.keys()) == {"A", "B"})
    for array in arrays.values():
        assert(array.shape == size)
