/**Copyright 2022 Moreno 

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and limitations under the License. **/

package org.testaiwolf.demo;

import java.util.ArrayList;
import java.util.List;

import org.aiwolf.client.lib.Content;
import org.aiwolf.common.data.Agent;
import org.aiwolf.common.data.Talk;
import org.aiwolf.common.net.GameInfo;
import org.aiwolf.common.net.GameSetting;

/**
 * Class for the possed role
 * @author Victor and Ruben
 */
public class Possessed extends BasePlayer {

	/**
	 * Getter of the name
	 */
	public String getName() {
		return "Possessed";
	}

	/**
	 * Initialize function
	 */
	public void initialize(GameInfo _gameInfo, GameSetting _gameSetting) {
		this.init(_gameInfo);
	}

	public void dayStart() {
	}

	/**
	 * Override of the updateBadguy function by selecting the agent with the highest trust coef instead of the lowest
	 */
	public void updateBadguy() {
		List<Agent> candidates = new ArrayList<>();
		for (Agent agent : getAliveAgentListNotMe()) {
			if (getTrustCoef(agent)>75) {
				candidates.add(agent);
			}
		}
		if (candidates.isEmpty()) {
			for (Agent agent : getAliveAgentListNotMe()) {
				if (getTrustCoef(agent)>50) {
					candidates.add(agent);
				}
			}
		}
		if (candidates.isEmpty()) {
			for (Agent agent : getAliveAgentListNotMe()) {
				if (getTrustCoef(agent)>25) {
					candidates.add(agent);
				}
			}
		}
		if (candidates.isEmpty()) {
			return;
		}
		badguy = randomSelect(candidates);
	}

	/**
	 * Update function, called on every action
	 */
	public void update(GameInfo gameInfo) {
		setCurrentGameInfo(gameInfo);
		// Extract CO, divination reports, and identification reports
		// from GameInfo.talkList
		for (int i = getTalkListHead(); i < getCurrentGameInfo().getTalkList().size(); i++) {
			Talk talk = getCurrentGameInfo().getTalkList().get(i);
			Agent talker = talk.getAgent();
			if (talker == getMe()) {
				continue;
			}
			Content content = new Content(talk.getText()); // Parse utterances
			switch (content.getTopic()) {
				case COMINGOUT:
					// Process CO
					getComingoutMap().put(talker, content.getRole()); // decrease trust coef if comingout is seer
					if (talk.getText().contains("MEDIUM"))
						changeTrustCoef(talker, 10);
					if (talk.getText().contains("SEER")){
						changeTrustCoef(talker, 10);
					}
					break;
				case DIVINED:
					if (talk.getText().contains("WEREWOLF")) {
						if (content.getTarget().equals(getMe()))
							changeTrustCoef(talker, -30);
						else
							changeTrustCoef(content.getTarget(), -10);
					} else {
						if (content.getTarget().equals(getMe()))
							changeTrustCoef(talker, 15);
						else
							changeTrustCoef(content.getTarget(), 10);
					}
					break;
				case IDENTIFIED: // Process identification report
					break;
				case VOTE:
					if(getCurrentGameInfo().getDay() > 5){
						if(getTrustCoef(talker) <  getTrustCoef(content.getTarget()))
							changeTrustCoef(talker, -10);
						else
							changeTrustCoef(content.getTarget(), -10);
					}
					break;
				default:
					break;
			}
		}
		setTalkListHead(getCurrentGameInfo().getTalkList().size());
	}

	/**
	 * Call for the Overrided villagerTalk function
	 */
	public String talk(){
		return villagerTalk();
    }

	public Agent attack() {
		throw new UnsupportedOperationException();
	}

	public String whisper() {
		throw new UnsupportedOperationException();
	}

	public Agent guard() {
		throw new UnsupportedOperationException();
	}

	public Agent divine() {
		throw new UnsupportedOperationException();
	}

	public void finish(){
		
	}
}
