import pandas as pd
import random
import os

# Set random seed for reproducibility
random.seed(42)

def create_realistic_sample():
    """Create a realistic-looking horse racing sample dataset by modifying the existing data."""
    # Check if the original CSV exists
    if not os.path.exists('raceform_2023_2025.csv'):
        print("Error: Original datafile raceform_2023_2025.csv not found.")
        return
    
    # Load a portion of the existing data (for speed)
    df = pd.read_csv('raceform_2023_2025.csv', nrows=10000)
    
    print(f"Loaded {len(df)} rows from original dataset.")
    
    # Real horse names
    horse_names = [
        "Flightline", "Life Is Good", "Epicenter", "Olympiad", "Taiba", "Cyberknife", "War Like Goddess", 
        "Jackie's Warrior", "Jack Christopher", "Letruska", "Nest", "Secret Oath", "Search Results",
        "Country Grammer", "Hot Rod Charlie", "Speaker's Corner", "Golden Pal", "Regal Glory", "Campanelle",
        "Clairiere", "Malathaat", "Matareya", "Shedaresthedevil", "Echo Zulu", "Just One Time", "Twilight Gleaming",
        "Obligatory", "Bella Sofia", "Ce Ce", "Goodnight Olive", "Kimari", "Gerrymander", "Jouster", "Cody's Wish",
        "Aloha West", "Mind Control", "Happy Saver", "Express Train", "Mandaloun", "Americanrevolution", "Mystic Guide", 
        "Max Player", "Proxy", "Lone Rock", "Fearless", "Colonel Liam", "Jupiter Inlet", "Classic Causeway", 
        "Magical", "Magic Wand", "Moonlight Magic", "Dawn Patrol", "Serpentine", "Fancy Blue", "Peaceful", "Siskin", 
        "Tarnawa", "Glass Slippers", "Teona", "Pretty Gorgeous", "Alcohol Free", "Mother Earth", "Dream of Dreams",
        "Battleground", "Order of Australia", "Thunder Moon", "Gear Up", "MacSwiney", "High Definition"
    ]
    
    # Real jockeys
    jockeys = [
        "Frankie Dettori", "Ryan Moore", "Oisin Murphy", "Jim Crowley", "William Buick", "James Doyle", 
        "Hollie Doyle", "Tom Marquand", "Harry Cobden", "Richard Johnson", "Bryony Frost", "Harry Skelton", 
        "Colin Keane", "Shane Foley", "Billy Lee", "Seamie Heffernan", "Wayne Lordan", "Paul Townend", 
        "Rachael Blackmore", "Davy Russell", "Danny Mullins", "Jack Kennedy", "Christophe Soumillon", 
        "Maxime Guyon", "Pierre-Charles Boudot", "Stephane Pasquier", "Mickael Barzalona", "Olivier Peslier", 
        "Irad Ortiz Jr.", "Joel Rosario", "John Velazquez", "Javier Castellano", "Flavien Prat", "Luis Saez", 
        "Jose Ortiz", "Tyler Gaffalione", "James McDonald", "Hugh Bowman", "Tommy Berry", "Damian Lane"
    ]
    
    # Real trainers
    trainers = [
        "John Gosden", "Sir Michael Stoute", "Aidan O'Brien", "Charlie Appleby", "Andrew Balding", 
        "William Haggas", "Roger Varian", "Nicky Henderson", "Paul Nicholls", "Dan Skelton", "Alan King", 
        "Philip Hobbs", "Willie Mullins", "Gordon Elliott", "Joseph O'Brien", "Jessica Harrington", 
        "Henry de Bromhead", "Dermot Weld", "Ger Lyons", "Donnacha O'Brien", "Tony Martin", "Gavin Cromwell", 
        "Andre Fabre", "Jean-Claude Rouget", "Francis-Henri Graffard", "Bob Baffert", "Todd Pletcher", 
        "Chad Brown", "Steve Asmussen", "Brad Cox", "Bill Mott", "Mark Casse", "Christophe Clement", 
        "Chris Waller", "James Cummings", "Ciaron Maher & David Eustace"
    ]
    
    # Real owners
    owners = [
        "Godolphin", "Juddmonte Farms", "Khalid Abdullah", "Hamdan Al Maktoum", "Qatar Racing", 
        "JP McManus", "Cheveley Park Stud", "King Power Racing", "Sheikh Mohammed Obaid", "Moyglare Stud", 
        "Gigginstown House Stud", "Mrs John Magnier", "Derrick Smith", "Michael Tabor", "Rich and Susannah Ricci", 
        "Middleham Park Racing", "Patricia Hunt", "Wertheimer et Frere", "Godolphin SNC", "H.H. Aga Khan", 
        "Winstar Farm", "Calumet Farm", "Stonestreet Stables", "China Horse Club", "Klaravich Stables", 
        "Coolmore", "James Harron Bloodstock", "Yulong Investments", "Arrowfield Stud"
    ]
    
    # Replace with realistic names
    print("Replacing with realistic horse names...")
    sample_horses = random.sample(horse_names, min(len(horse_names), len(df['horse_name'].unique())))
    horse_name_map = dict(zip(df['horse_name'].unique(), sample_horses + [f"Horse {i+1000}" for i in range(len(df['horse_name'].unique()) - len(sample_horses))]))
    df['horse_name'] = df['horse_name'].map(horse_name_map)
    
    print("Replacing with realistic jockey names...")
    sample_jockeys = random.sample(jockeys, min(len(jockeys), len(df['jockey'].unique())))
    jockey_map = dict(zip(df['jockey'].unique(), sample_jockeys + [f"Jockey {i+1000}" for i in range(len(df['jockey'].unique()) - len(sample_jockeys))]))
    df['jockey'] = df['jockey'].map(jockey_map)
    
    print("Replacing with realistic trainer names...")
    sample_trainers = random.sample(trainers, min(len(trainers), len(df['trainer'].unique())))
    trainer_map = dict(zip(df['trainer'].unique(), sample_trainers + [f"Trainer {i+1000}" for i in range(len(df['trainer'].unique()) - len(sample_trainers))]))
    df['trainer'] = df['trainer'].map(trainer_map)
    
    print("Replacing with realistic owner names...")
    sample_owners = random.sample(owners, min(len(owners), len(df['owner'].unique())))
    owner_map = dict(zip(df['owner'].unique(), sample_owners + [f"Owner {i+1000}" for i in range(len(df['owner'].unique()) - len(sample_owners))]))
    df['owner'] = df['owner'].map(owner_map)
    
    # Save as new CSV and Excel files
    df.to_csv('realistic_racing_data.csv', index=False)
    print(f"Created CSV file with {len(df)} rows of realistic-looking data.")
    
    # Save subset to Excel (Excel has row limitations)
    excel_rows = min(len(df), 50000)
    df.head(excel_rows).to_excel('realistic_racing_data.xlsx', index=False)
    print(f"Created Excel file with {excel_rows} rows of realistic-looking data.")

if __name__ == "__main__":
    create_realistic_sample()