from pypop import World, Species, Occupant, BirthReaction, DeathReaction, PredationReaction, PredationBirthReaction, Hop
from pypop.pypop import Site, Lattice, Reaction, SiteFullException

import numpy as np
import pytest
from pytest import raises
from contextlib import nullcontext as does_not_raise

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

def _binomial_expectation(successes, trials, rate):
    mean = trials*rate
    stddev = trials * rate * (1-rate)
    return successes > (mean - 2*stddev) and successes < (mean + 2*stddev)

@pytest.mark.parametrize('carrying_capacity', [None, 1])
def test_reactions(carrying_capacity):
    # Create species, site and occupant to be used below
    species = Species("A")
    lattice = Lattice((2, 2), carrying_capacity=carrying_capacity)
    site = lattice.sites[0]
    other_site = lattice.sites[1]
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
    assert(_binomial_expectation(successes, trials, rate))

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
    assert(_binomial_expectation(successes, trials, rate))

    # Test PredationReaction initialization
    prey_species = Species("B")
    predation_reaction = PredationReaction(species, prey_species, rate)
    assert(predation_reaction.speciesA == species)
    assert(predation_reaction.speciesB == prey_species)
    assert(predation_reaction.rate == rate)
    # Test that the reaction happens at the correct rate
    # Note, this can fail randomly, but should be fine mostly!! It is a statistical test after all.
    if carrying_capacity == 1:
        successes = 0
        for _ in range(trials):
            prey = Occupant(prey_species, other_site)
            assert(predation_reaction(occupant, rng) == False)
            if len(prey_species) == 0:
                successes += 1
            else:
                prey.destroy()
    else:
        for _ in range(trials):
            Occupant(prey_species, site)
        assert(predation_reaction(occupant, rng) == False)
        successes = trials - len(prey_species)
    assert(_binomial_expectation(successes, trials, rate))

    # Test PredationBirthReaction initialization
    prey_species = Species("C")
    predationbirth_reaction = PredationBirthReaction(species, prey_species, rate)
    assert(predationbirth_reaction.speciesA == species)
    assert(predationbirth_reaction.speciesB == prey_species)
    assert(predationbirth_reaction.rate == rate)
    # Test that the reaction happens at the correct rate
    # Note, this can fail randomly, but should be fine mostly!! It is a statistical test after all.
    if carrying_capacity == 1:
        successes = 0
        for _ in range(trials):
            prey = Occupant(prey_species, other_site)
            assert(predationbirth_reaction(occupant, rng) == False)
            if len(prey_species) == 0 and len(other_site.species_occupants[species]) == 1:
                successes += 1
                other_site.species_occupants[species][0].destroy()
            else:
                prey.destroy()
    else:
        for _ in range(trials):
            Occupant(prey_species, site)
        assert(predationbirth_reaction(occupant, rng) == False)
        successes = trials - len(prey_species)
        assert(len(species) == successes + 1)
    assert(_binomial_expectation(successes, trials, rate))

def test_birthreaction_singleoccupancy():
    species = Species("A")
    lattice = Lattice((2, 2), carrying_capacity=1)
    for site in lattice.sites:
        Occupant(species, site)
    
    rng = np.random.default_rng()
    BirthReaction(species, 1.0)(lattice.sites[0].species_occupants[species][0], rng)
    assert(len(species) == 4)

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

def test_carrying_capacity():
    species = Species("A")
    lattice = Lattice((1, 2), carrying_capacity=2)
    site = lattice.sites[0]
    second_site = lattice.sites[1]

    rng = np.random.default_rng()

    occupant1 = Occupant(species, site)
    occupant2 = Occupant(species, site)
    with raises(SiteFullException):
        Occupant(species, site)

    occupant3 = Occupant(species, second_site)
    # with raises(SiteFullException):
    #     occupant3.set_site(site)
    
    Hop(species, 1.0)(occupant3, rng)
    assert(occupant3.site == second_site)
    
    occupant3.destroy()

    BirthReaction(species, 1.0)(occupant1, rng)
    assert(len(species) == 2)
    occupant2.destroy()

    prey_species = Species("B")
    Occupant(prey_species, site)
    PredationBirthReaction(species, prey_species, 1.0)(occupant1, rng)
    # Because PredationBirthReaction kills one particle and creates another, SiteFullException is never raised.
    assert(len(species) == 2)
    assert(len(prey_species) == 0)

def test_swap_sites():
    species = Species("A")
    lattice = Lattice((1, 2), carrying_capacity=1)

    first_site = lattice.sites[0]
    second_site = lattice.sites[1]

    first_site.neighbors = [second_site]
    second_site.neighbors = [first_site]

    rng = np.random.default_rng()
    hop = Hop(species, 1.0)

    occupant1 = Occupant(species, first_site)
    occupant2 = Occupant(species, second_site)

    hop(occupant1, rng)
    assert(occupant1.site == second_site)
    assert(occupant2.site == first_site)

    occupant2.destroy()
    hop(occupant1, rng)

@pytest.mark.parametrize(
    'carrying_capacity,density,expectation',
    [
        (None, 10.0, does_not_raise()),
        (1, 0.5, does_not_raise()),
        (1, 10.0, raises(ValueError, match="Cannot place particles since total density of all species exceeds the carrying capacity."))
    ]
)
def test_world(carrying_capacity, density, expectation):
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
    with expectation:
        world = World(
            size=size,
            initial_densities={A: density, B: density},
            hops={A: Hop(A, 1.0), B: Hop(A, 1.0)},
            reactions=reactions,
            carrying_capacity=carrying_capacity,
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
