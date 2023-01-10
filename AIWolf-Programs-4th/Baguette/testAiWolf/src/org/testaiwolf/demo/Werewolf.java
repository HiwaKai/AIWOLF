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
import org.aiwolf.client.lib.ContentBuilder;
import org.aiwolf.client.lib.VoteContentBuilder;
import org.aiwolf.common.data.Agent;
import org.aiwolf.common.data.Talk;
import org.aiwolf.common.net.GameInfo;
import org.aiwolf.common.net.GameSetting;

/**
 * Class for the werewolf role
 * @author Victor and Ruben
 */
public class Werewolf extends BasePlayer {

	/** Current target of the attack */
	Agent target;
	/** Last target of the attack */
	Agent lastTarget;

	/**
	 * Getter of the name
	 */
	public String getName() {
		return "Werewolf";
	}

	/**
	 * Initialize function
	 */
	public void initialize(GameInfo _gameInfo, GameSetting _gameSetting) {
		this.init(_gameInfo);
	}

	public void dayStart() {
		return;
	}

	/**
	 * Override of the updateBadguy function by selecting an agent with a trust coef higher than 50
	 */
	public void updateBadguy() {
		List<Agent> candidates = new ArrayList<>();

		if (candidates.isEmpty()) {
			for (Agent agent : getAliveAgentListNotMe()) {
				if (getTrustCoef(agent) > 50) {
					candidates.add(agent);
				}
			}
		}
		if (candidates.isEmpty()) {
			for (Agent agent : getAliveAgentListNotMe()) {
				if (getTrustCoef(agent) > 25) {
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
		// Extract CO, divination reports, and identification reports from GameInfo.talkList
		for (int i = getTalkListHead(); i < getCurrentGameInfo().getTalkList().size(); i++) {
			Talk talk = getCurrentGameInfo().getTalkList().get(i);
			Agent talker = talk.getAgent();
			if (getCurrentGameInfo().getRoleMap().keySet().contains(talker)) { //ignore messages sent by werewolf
				continue;
			}
			Content content = new Content(talk.getText()); // Parse utterances
			switch (content.getTopic()) {
				case COMINGOUT:
					// Process CO
					getComingoutMap().put(talker, content.getRole()); // decrease trust coef if comingout is seer
					if (talk.getText().contains("MEDIUM") || talk.getText().contains("SEER"))
						changeTrustCoef(talker, 15);
					break;
				case DIVINED:
					if (talk.getText().contains("WEREWOLF")) {
						if (getCurrentGameInfo().getRoleMap().keySet().contains(content.getTarget()))
							changeTrustCoef(talker, 30);
					} else {
						if (!getCurrentGameInfo().getRoleMap().keySet().contains(content.getTarget()))
							changeTrustCoef(content.getTarget(), 10);
						else
							setTrustCoef(talker, 0);
					}
					break;
				case VOTE:
					if (getCurrentGameInfo().getRoleMap().keySet().contains(content.getTarget())){
						changeTrustCoef(talker, 20);
					}
					break;
				case IDENTIFIED: // Process identification report
					break;
				default:
					break;
			}
		}
		setTalkListHead(getCurrentGameInfo().getTalkList().size());
	}

	/**
	 * Call of the Overrided villagerTalk function
	 */
	public String talk() {
		return villagerTalk();
	}

	/**
	 * Select a target with the highest trust coef and not the same as the previous night if it survived
	 */
	public void selectTarget() {
		lastTarget = target;
		target = null;
		for (Agent agent : getAliveAgentListNotMe()) {
			if ((target == null || getTrustCoef(agent)>getTrustCoef(target)) && agent != lastTarget) {
				target = agent;
			}
		}
	}

	/**
	 * Attack the pre selected target
	 */
	public Agent attack() {
		return this.target;
	}

	/**
	 * Before attacking, announce to the other werewolf what it's next target will be
	 */
	public String whisper() {
		selectTarget();
        ContentBuilder builder = new VoteContentBuilder(this.target);
				return new Content(builder).getText();
	}

	public Agent guard() {
		throw new UnsupportedOperationException();
	}

	public Agent divine() {
		throw new UnsupportedOperationException();
	}

	public void finish() {
		
	}
}
