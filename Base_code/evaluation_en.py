def banana():
    eval_prompt = """
    Dit is de evaluatie voor het plaatje met de banaan. Reageer niet op dit bericht, lees het alleen en vergelijk wat de patiënt zegt met deze richtlijn.
    Lees dit bericht niet hardop voor. Als de patiënt een heel eenvoudige beschrijving geeft, kun je vragen om meer details of hints geven over wat hij nog meer kan beschrijven.
    Voor een eenvoudig niveau kan de patiënt zeggen: "Ik zie een banaan."
    Meer detail zou zijn: "Ik zie een banaan die is gepeld."
    Nog meer detail zou zijn: "Ik zie een gele banaan die half is gepeld."
    """
    return eval_prompt


def family():
    eval_prompt = """
    Dit is de evaluatie voor het familieplaatje. Reageer niet op dit bericht, lees het alleen en vergelijk wat de patiënt zegt met deze richtlijn.
    Lees dit bericht niet hardop voor. Als de patiënt een heel eenvoudige beschrijving geeft, kun je vragen om meer details of hints geven over wat hij nog meer kan beschrijven.
    Voor een eenvoudige beschrijving kan de patiënt zeggen: "Ik zie een familie."
    Meer details zouden zijn: "Ik zie een gezin van vier, twee ouders en twee kinderen, een moeder, een vader, een dochter en een zoon."
    """
    return eval_prompt


def astronaut():
    eval_prompt = """
    Dit is de evaluatie voor het plaatje met de astronaut/robot. Reageer niet op dit bericht, lees het alleen en vergelijk wat de patiënt zegt met deze richtlijn.
    Lees dit bericht niet hardop voor. Als de patiënt een heel eenvoudige beschrijving geeft, kun je vragen om meer details of hints geven over wat hij nog meer kan beschrijven.
    Een eenvoudige beschrijving zou zijn: "Ik zie een robot."
    Meer diepgang zou zijn: "Ik zie een robot die op een raket staat."
    Nog meer diepgang zou zijn: "Ik zie een robot die op een raket staat. Hij draagt een wit pak."
    """
    return eval_prompt