import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create a sample dataset with race data from 2023-2025
def create_race_data():
    """Create a sample horse racing dataset for 2023-2025."""
    # Configuration
    num_races = 500  # Reduced from 5000 for faster processing
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 4, 30)
    
    # Race parameters
    race_types = ['Flat', 'Hurdle', 'Chase', 'National Hunt Flat', 'Stakes', 'Handicap', 'Maiden', 'Novice']
    race_classes = ['Group 1', 'Group 2', 'Group 3', 'Listed', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']
    race_distances = ['5f', '6f', '7f', '1m', '1m 2f', '1m 4f', '1m 6f', '2m', '2m 2f', '2m 4f', '2m 6f', '3m', '3m 2f', '4m']
    going_conditions = ['Firm', 'Good to Firm', 'Good', 'Good to Soft', 'Soft', 'Heavy', 'Standard', 'Slow', 'Standard to Slow']
    
    # Real racecourses by country
    racecourses_by_country = {
        'UK': ['Ascot', 'Newmarket', 'Cheltenham', 'Aintree', 'York', 'Goodwood', 'Epsom', 'Doncaster', 'Newbury', 'Haydock', 
               'Kempton', 'Sandown', 'Lingfield', 'Leicester', 'Wolverhampton', 'Newcastle', 'Windsor', 'Ayr', 'Musselburgh'],
        'Ireland': ['Leopardstown', 'Curragh', 'Punchestown', 'Fairyhouse', 'Galway', 'Naas', 'Navan', 'Tipperary', 'Cork', 'Limerick'],
        'France': ['Longchamp', 'Chantilly', 'Deauville', 'Saint-Cloud', 'Auteuil', 'Maisons-Laffitte', 'Cagnes-sur-Mer'],
        'USA': ['Churchill Downs', 'Belmont Park', 'Saratoga', 'Del Mar', 'Santa Anita', 'Pimlico', 'Aqueduct', 'Keeneland', 'Gulfstream Park'],
        'Australia': ['Flemington', 'Randwick', 'Caulfield', 'Moonee Valley', 'Rosehill', 'Eagle Farm', 'Doomben']
    }
    
    countries = list(racecourses_by_country.keys())
    
    # Real jockeys by country
    jockeys_by_country = {
        'UK': ['Frankie Dettori', 'Ryan Moore', 'Oisin Murphy', 'Jim Crowley', 'William Buick', 'James Doyle', 'Hollie Doyle',
               'Tom Marquand', 'Harry Cobden', 'Richard Johnson', 'Bryony Frost', 'Harry Skelton', 'Sam Twiston-Davies'],
        'Ireland': ['Colin Keane', 'Shane Foley', 'Billy Lee', 'Seamie Heffernan', 'Wayne Lordan', 'Paul Townend', 'Rachael Blackmore', 
                   'Davy Russell', 'Danny Mullins', 'Jack Kennedy'],
        'France': ['Christophe Soumillon', 'Maxime Guyon', 'Pierre-Charles Boudot', 'Stephane Pasquier', 'Mickael Barzalona', 'Olivier Peslier'],
        'USA': ['Irad Ortiz Jr.', 'Joel Rosario', 'John Velazquez', 'Javier Castellano', 'Flavien Prat', 'Luis Saez', 'Jose Ortiz', 'Tyler Gaffalione'],
        'Australia': ['James McDonald', 'Hugh Bowman', 'Tommy Berry', 'Damian Lane', 'Glen Boss', 'Craig Williams', 'Kerrin McEvoy', 'Nash Rawiller']
    }
    
    # Real trainers by country
    trainers_by_country = {
        'UK': ['John Gosden', 'Sir Michael Stoute', 'Aidan O\'Brien', 'Charlie Appleby', 'Andrew Balding', 'William Haggas', 
              'Roger Varian', 'Nicky Henderson', 'Paul Nicholls', 'Dan Skelton', 'Alan King', 'Philip Hobbs'],
        'Ireland': ['Willie Mullins', 'Gordon Elliott', 'Joseph O\'Brien', 'Jessica Harrington', 'Henry de Bromhead', 'Dermot Weld', 
                   'Ger Lyons', 'Donnacha O\'Brien', 'Tony Martin', 'Gavin Cromwell'],
        'France': ['Andre Fabre', 'Jean-Claude Rouget', 'Francis-Henri Graffard', 'Alain de Royer-Dupre', 'Carlos Laffon-Parias'],
        'USA': ['Bob Baffert', 'Todd Pletcher', 'Chad Brown', 'Steve Asmussen', 'Brad Cox', 'Bill Mott', 'Mark Casse', 'Christophe Clement'],
        'Australia': ['Chris Waller', 'James Cummings', 'Ciaron Maher & David Eustace', 'Gai Waterhouse & Adrian Bott', 'Anthony Freedman']
    }
    
    # Real owners by country
    owners_by_country = {
        'UK': ['Godolphin', 'Juddmonte Farms', 'Khalid Abdullah', 'Hamdan Al Maktoum', 'Qatar Racing', 'JP McManus', 'Cheveley Park Stud', 
              'King Power Racing', 'Sheikh Mohammed Obaid', 'Moyglare Stud'],
        'Ireland': ['Gigginstown House Stud', 'Mrs John Magnier', 'Derrick Smith', 'Michael Tabor', 'Rich and Susannah Ricci', 
                    'JP McManus', 'Middleham Park Racing', 'Patricia Hunt'],
        'France': ['Wertheimer et Frere', 'Godolphin SNC', 'H.H. Aga Khan', 'Ecurie Jean-Louis Bouchard', 'Haras de Saint Pair'],
        'USA': ['Godolphin', 'Winstar Farm', 'Juddmonte Farms', 'Calumet Farm', 'Stonestreet Stables', 'China Horse Club', 'Klaravich Stables'],
        'Australia': ['Godolphin', 'Coolmore', 'James Harron Bloodstock', 'Aquis Farm', 'Yulong Investments', 'Arrowfield Stud']
    }
    
    # Creative horse names with real-world patterns
    first_parts = ['Royal', 'Golden', 'Silver', 'Midnight', 'Highland', 'Northern', 'Southern', 'Eastern', 'Western', 'Brave', 
                  'Swift', 'Wild', 'Noble', 'Proud', 'Quiet', 'Silent', 'Lucky', 'Magic', 'Mystic', 'Mighty', 'Fast', 'Dark', 
                  'Bright', 'Red', 'Blue', 'Green', 'Black', 'White', 'Captain', 'Lord', 'Lady', 'Master', 'Sir', 'Prince', 'Princess',
                  'King', 'Queen', 'Emperor', 'Empress', 'Duke', 'Duchess']
    
    second_parts = ['Star', 'Moon', 'Sun', 'Light', 'Shadow', 'Runner', 'Dancer', 'Jumper', 'Flyer', 'Thunder', 'Lightning', 'Storm', 
                   'Wind', 'River', 'Ocean', 'Mountain', 'Valley', 'Forest', 'Meadow', 'Field', 'Sky', 'Cloud', 'Rain', 'Snow', 
                   'Fire', 'Ice', 'Diamond', 'Gold', 'Silver', 'Legend', 'Dream', 'Whisper', 'Song', 'Tale', 'Heart', 'Spirit', 
                   'Soul', 'Hope', 'Wish', 'Victory', 'Glory', 'Honor', 'Pride', 'Joy', 'Wonder', 'Magic', 'Charm', 'Beauty']
    
    # Generate horse names
    horse_names = []
    for _ in range(500):
        name_type = random.randint(1, 4)
        if name_type == 1:
            # Two-word name
            name = f"{random.choice(first_parts)} {random.choice(second_parts)}"
        elif name_type == 2:
            # Single word with apostrophe
            name = f"{random.choice(first_parts)}'s {random.choice(second_parts)}"
        elif name_type == 3:
            # Single word with 'The'
            name = f"The {random.choice(first_parts)} {random.choice(second_parts)}"
        else:
            # Random name from real-world patterns
            special_names = [
                "Seabiscuit", "Man o' War", "Secretariat", "Seattle Slew", "Affirmed", "Justify", "American Pharoah",
                "Cigar", "Zenyatta", "Winx", "Black Caviar", "Frankel", "Enable", "Goldikova", "Kauto Star", "Desert Orchid",
                "Red Rum", "Tiger Roll", "Best Mate", "Denman", "Altior", "Big Buck's", "Dawn Run", "Istabraq", "Hurricane Fly",
                "Nijinsky", "Mill Reef", "Brigadier Gerard", "Dancing Brave", "Nashwan", "Montjeu", "Galileo", "Sea The Stars",
                "Deep Impact", "Harbinger", "El Condor Pasa", "A.P. Indy", "Sunday Silence", "Deep Impact", "Dubawi", "Danehill",
                "Sadler's Wells", "Northern Dancer", "Hyperion", "Ribot", "Nearco", "St. Simon", "Eclipse", "Kauto Star",
                "Denman", "Sprinter Sacre", "Big Buck's", "Hurricane Fly", "Annie Power", "Faugheen", "Sizing John"
            ]
            name = random.choice(special_names)
            # Ensure no duplicates
            while name in horse_names:
                name = random.choice(special_names)
        
        horse_names.append(name)
        # Ensure no duplicates
        if len(set(horse_names)) < len(horse_names):
            horse_names = list(set(horse_names))
            horse_names.append(f"{random.choice(first_parts)} {random.choice(second_parts)}")
    
    ages = list(range(2, 13))
    ratings = list(range(50, 120))
    weights = [str(8 + i//2) + '-' + str(random.randint(0, 13)) for i in range(20)]
    
    # Generate random dates between start and end
    days_range = (end_date - start_date).days
    race_dates = [start_date + timedelta(days=random.randint(0, days_range)) for _ in range(num_races)]
    race_dates.sort()  # Sort chronologically
    
    # Create data for each race
    data = []
    race_id = 1000
    
    for race_date in race_dates:
        race_id += 1
        
        # Select country and then a specific racecourse in that country
        country = random.choice(countries)
        racecourse = random.choice(racecourses_by_country[country])
        
        race_type = random.choice(race_types)
        race_class = random.choice(race_classes)
        race_distance = random.choice(race_distances)
        going = random.choice(going_conditions)
        
        # Adjust prize money based on race class
        if 'Group 1' in race_class:
            prize_money = random.randint(250000, 1000000)
        elif 'Group 2' in race_class:
            prize_money = random.randint(150000, 300000)
        elif 'Group 3' in race_class:
            prize_money = random.randint(75000, 150000)
        elif 'Listed' in race_class or 'Class 1' in race_class:
            prize_money = random.randint(40000, 80000)
        else:
            prize_money = random.randint(5000, 40000)
        
        # Number of horses in this race
        num_horses = random.randint(5, 16)
        
        # Create finishing positions
        positions = list(range(1, num_horses + 1))
        
        # Select a set of horses for this race (no duplicates within race)
        race_horses = random.sample(horse_names, num_horses)
        
        # Is this a handicap race?
        is_handicap = 'Handicap' in race_type or random.random() < 0.4
        
        # Create entries for each horse
        for position, horse_name in zip(positions, race_horses):
            # Select jockey and trainer from the right country
            jockey = random.choice(jockeys_by_country[country])
            trainer = random.choice(trainers_by_country[country])
            owner = random.choice(owners_by_country[country])
            
            age = random.choice(ages)
            
            # Adjust ratings based on race class
            if 'Group 1' in race_class or 'Class 1' in race_class:
                rating = random.randint(90, 120)
            elif 'Group 2' in race_class or 'Group 3' in race_class or 'Class 2' in race_class:
                rating = random.randint(80, 100)
            else:
                rating = random.randint(50, 85)
            
            weight = random.choice(weights)
            
            # Add some performance metrics
            if position == 1:
                beat_favorite = random.choice([True, False])
                sp_odds = random.randint(1, 8) if random.random() < 0.7 else random.randint(9, 20)
            elif position == 2:
                beat_favorite = position == 1
                sp_odds = random.randint(2, 12) if random.random() < 0.7 else random.randint(13, 25)
            else:
                beat_favorite = False
                sp_odds = random.randint(3, 50)
            
            # Calculate finish time based on distance
            base_minutes = 1
            if '3m' in race_distance or '4m' in race_distance:
                base_minutes = 6
            elif '2m' in race_distance:
                base_minutes = 3
            elif '1m' in race_distance:
                base_minutes = 1
            
            # Add some variability based on position
            position_adjustment = (position - 1) * 0.5  # Each position adds half a second
            
            finish_time = timedelta(
                minutes=base_minutes, 
                seconds=random.randint(20, 59) + position_adjustment,
                milliseconds=random.randint(0, 999)
            )
            
            # Last 3 race positions (form)
            last_three_form = ''.join([str(random.randint(1, 9)) for _ in range(3)])
            
            # Occasionally add P (pulled up), F (fell), U (unseated), or R (refused) for jump races
            if race_type in ['Hurdle', 'Chase'] and random.random() < 0.15 and position > num_horses//2:
                last_three_form = list(last_three_form)
                idx = random.randint(0, 2)
                last_three_form[idx] = random.choice(['P', 'F', 'U', 'R'])
                last_three_form = ''.join(last_three_form)
            
            data.append({
                'race_id': race_id,
                'race_date': race_date,
                'racecourse': racecourse,
                'country': country,
                'race_type': race_type,
                'race_class': race_class,
                'race_distance': race_distance,
                'going': going,
                'prize_money': prize_money,
                'position': position,
                'horse_name': horse_name,
                'jockey': jockey,
                'trainer': trainer,
                'owner': owner,
                'age': age,
                'official_rating': rating,
                'weight_carried': weight,
                'sp_odds': sp_odds,
                'beat_favorite': beat_favorite,
                'finish_time': str(finish_time),
                'form': last_three_form,
                'year': race_date.year,
                'month': race_date.month,
                'handicap': is_handicap
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Additional calculated fields
    df['days_since_last_run'] = np.random.randint(7, 100, size=len(df))
    df['ground_preference'] = np.random.choice(['Prefers Firm', 'Prefers Good', 'Prefers Soft', 'Versatile'], size=len(df))
    
    # Format fractional odds in realistic formats
    def format_odds(odds):
        if odds <= 2:
            return f"{odds}/1"
        elif odds <= 4:
            return f"{odds-1}/1"
        elif odds == 5:
            return "4/1"
        elif odds == 6:
            return "5/1"
        elif odds == 7:
            return "6/1"
        elif odds == 8:
            return "7/1"
        elif odds == 9:
            return "8/1"
        elif odds == 10:
            return "9/1"
        elif odds <= 12:
            return "10/1"
        elif odds <= 14:
            return "12/1"
        elif odds <= 16:
            return "14/1"
        elif odds <= 20:
            return "16/1"
        elif odds <= 25:
            return "20/1"
        elif odds <= 33:
            return "25/1"
        elif odds <= 50:
            return "33/1"
        else:
            return "50/1"
    
    df['fractional_odds'] = df['sp_odds'].apply(format_odds)
    
    # Add performance trends
    df['win_percentage'] = np.random.randint(0, 41, size=len(df))
    df['place_percentage'] = df['win_percentage'] + np.random.randint(0, 30, size=len(df))
    df['place_percentage'] = df['place_percentage'].clip(upper=100)
    
    return df

def main():
    print("Generating sample racing dataset for 2023-2025...")
    race_data = create_race_data()
    
    # Save as CSV
    race_data.to_csv('raceform_2023_2025.csv', index=False)
    print(f"CSV saved with {len(race_data)} records.")
    
    # Save as Excel if not too large
    if len(race_data) <= 1048576:  # Excel row limit
        race_data.to_excel('raceform_2023_2025.xlsx', index=False)
        print("Excel file also saved.")
    else:
        # Save a subset if too large
        max_rows = 1000000
        race_data.iloc[:max_rows].to_excel('raceform_2023_2025.xlsx', index=False)
        print(f"Excel file saved with {max_rows} records (Excel row limit).")

if __name__ == "__main__":
    main()