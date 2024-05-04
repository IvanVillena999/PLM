#Create Employee Profile
profile = {'gender': ['male', 'female'], 'position': ['waiter', 'waitress'], 'stations': ['lobby', 'vip']}

#OT Rates
min_wage = 610
ot_rate = 0.30
ot_pay = min_wage * ot_rate

#Compute for the OT payment
fred_ot = [2, 3, 5, 2, 5]
fred_pay = sum(fred_ot) * ot_pay


#Summarize the Profile
print(f"Fred is a {profile['gender'][0]} {profile['position']} assigned in {profile['stations'][0]} "
      "and received an overtime pay of", fred_pay)
