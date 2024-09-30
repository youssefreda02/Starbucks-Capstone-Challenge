def find_offer(offer_id, portfolio_clean):
    '''
        This function finds the offer by its id and return the whole offer row
        
        input: offer_id (String)
        
        return: (Pandans DataFrame) that contains 1 row.
    '''
    return portfolio_clean[portfolio_clean['id']== offer_id]

def find_offer_id(user_id, transcript_clean):
    '''
        This function finds the offers from the events that made by the user_id.
        Returns these offers only from the event.
        
        input: user_id (String)
        
        return: user_events (Pandans DataFrame) 
    '''
    # Retrieving the user events
    user_events = transcript_clean[transcript_clean['person'] == user_id].sort_values('time')
    
    # Returning only the events that are offers related not transactions
    return user_events[user_events['offer_id'] != 0.0][['offer_id','event']]

def complete_without_view(user_id, transcript_clean):
    '''
        This function takes a user id, and searches for offers that he completed without receiving it.

        input: user_id (String)

        return: not_viewed (List)
    '''

    # Retrieving offers that related to user_id
    offers = find_offer_id(user_id,transcript_clean)

    #init
    not_viewed = []
    #looping through offers events
    for _, offer in offers.iterrows():
        
        qu = offers[offers['offer_id'] == offer.iloc[0]]
        
        i = 0
        for _, of in qu.iterrows():
            
            if (of['event'] == 'offer viewed'):
                i = i-1
                continue
                
            if (of['event'] == 'offer completed'):
                i = i+1
                continue
                
        if i > 0:
            not_viewed.append(offer.iloc[0])
            break
            
        offers = offers[offers['offer_id']!= offer.iloc[0]]
        
    return not_viewed


def view_without_complete (user_id, transcript_clean):
    '''
        This function takes user id and returns a list of offers that being viewed but not completed

        input: user_id (String)

        return: not_completed (List)
    '''
    offers = find_offer_id(user_id, transcript_clean)
    
    not_completed = []
    
    for _, offer in offers.iterrows():
        
        qu = offers[offers['offer_id'] == offer.iloc[0]]
        
        i = 0
        for _, of in qu.iterrows():
            
            if (of['event'] == 'offer viewed'):
                i = i + 1
                continue
                
            if (of['event'] == 'offer completed'):
                i = i - 1
                continue
        
        if i > 0:
            not_completed.append(offer.iloc[0])
            
        offers = offers[offers['offer_id']!= offer.iloc[0]]
        
    return not_completed


def complete_after_view (user_id, transcript_clean):
    
    '''
        This function takes user id and returns a list of offers that being viewed but not completed

        input: user_id (String)

        return: not_completed (List)
    '''
    # Retrieving offers that related to user_id
    offers = find_offer_id(user_id, transcript_clean)

    #init
    completed = []
    #looping through offers events
    for _, offer in offers.iterrows():
        
        qu = offers[offers['offer_id'] == offer.iloc[0]]
        
        i = 0 # to check state
        n = 0 # count how many
        for _, of in qu.iterrows():
            
            if (i%3 == 0) and (of['event'] == 'offer received'):
                i = 1
                continue
                
            if (i == 1) and (of['event'] == 'offer viewed'):
                i = 2
                continue
                
            if (i== 2) and (of['event'] == 'offer completed'):
                i = 3
                n = n+1
                continue
            
        if i == 3:
            for _ in range(n):
                completed.append(offer.iloc[0])
            
        offers = offers[offers['offer_id']!= offer.iloc[0]]
        
    return completed




def user_offers_response(user_id, transcript_clean):
    """
        Summarizes the response of a user to different offers.
        This function takes a user ID and returns a dictionary summarizing the user's interaction with offers.
        The summary includes:
        - Offers that were viewed and then completed.
        - Offers that were viewed but not completed.
        - Offers that were completed without being viewed.
    
        Args:
            user_id (str): The ID of the user.
    
        Returns: (dict): A dictionary with the following keys:
                - 'complete_after_view' (list): List of offer IDs that were viewed and then completed by the user.
                - 'view_without_complete' (list): List of offer IDs that were viewed but not completed by the user.
                - 'complete_without_view' (list): List of offer IDs that were completed without being viewed by the user.
    """
    return {
        'complete_after_view': complete_after_view(user_id, transcript_clean),
        'view_without_complete': view_without_complete(user_id, transcript_clean),
        'complete_without_view': complete_without_view(user_id, transcript_clean)
    }