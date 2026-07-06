from transitions import Machine

class PokerStateMachine:
    states = [
        'IDLE',                 # Bot is inactive
        'WAITING_FOR_TURN',     # Scanning the screen every 1s, waiting for fold button to appear
        'READING_STATE',        # Fold button found. Extracting cards, pot, stacks, etc.
        'DECIDING',             # Running hand evaluation and calculating decision
        'EXECUTING_ACTION'      # Performing clicks/mouse movements
    ]

    def __init__(self, callback_handler=None):
        self.callback_handler = callback_handler # Receives updates for logging/GUI
        
        # Initialize state machine
        self.machine = Machine(
            model=self, 
            states=PokerStateMachine.states, 
            initial='IDLE',
            send_event=True,
            ignore_invalid_triggers=True
        )

        # Define transitions
        # format: trigger, source, dest
        self.machine.add_transition('start', 'IDLE', 'WAITING_FOR_TURN', after='on_state_change')
        self.machine.add_transition('stop', '*', 'IDLE', after='on_state_change')
        
        self.machine.add_transition('turn_detected', 'WAITING_FOR_TURN', 'READING_STATE', after='on_state_change')
        self.machine.add_transition('state_read_complete', 'READING_STATE', 'DECIDING', after='on_state_change')
        self.machine.add_transition('decision_made', 'DECIDING', 'EXECUTING_ACTION', after='on_state_change')
        self.machine.add_transition('action_completed', 'EXECUTING_ACTION', 'WAITING_FOR_TURN', after='on_state_change')
        
        # Fail-safe transition back to WAITING_FOR_TURN on error
        self.machine.add_transition('error_occurred', '*', 'WAITING_FOR_TURN', after='on_state_change')

    def on_state_change(self, event):
        """Callback invoked after every state transition."""
        state_name = self.state
        print(f"[State Machine] Transitioned to: {state_name}")
        if self.callback_handler and hasattr(self.callback_handler, 'on_state_updated'):
            self.callback_handler.on_state_updated(state_name)

if __name__ == '__main__':
    # Simple test run of state machine
    class DummyHandler:
        def on_state_updated(self, state):
            print(f"  --> GUI Handler received state update: {state}")
            
    handler = DummyHandler()
    sm = PokerStateMachine(handler)
    
    print(f"Initial State: {sm.state}")
    sm.start()
    sm.turn_detected()
    sm.state_read_complete()
    sm.decision_made()
    sm.action_completed()
    sm.stop()
