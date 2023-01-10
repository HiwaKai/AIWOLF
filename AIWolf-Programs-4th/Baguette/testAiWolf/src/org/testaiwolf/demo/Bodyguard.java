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
 * Bodyguard class for the role of Bodydguard
 * @author Victor & Ruben
 */
public class Bodyguard extends BasePlayer {

	/** Last agent protected */
	Agent lastProtected;
	/** Previously attacked (and killed) agent */
	Agent previouslyAttacked;
	/** Boolean to see if an agent died last night */
	boolean noDeath;

	/**  
	 * Getter of the name
	 */
	public String getName() {
		return "Bodyguard";
	}

	/**
	 * Initialize function
	 */
	public void initialize(GameInfo _gameInfo, GameSetting _gameSetting) {
		this.init(_gameInfo);
		lastProtected = null;
		previouslyAttacked = null;
		noDeath = true;
	}

	/**
	 * Function called upon the start of the day
	 * Used to improve the bodyguard guarding
	 */
	public void dayStart() {
		if(previouslyAttacked == null && getCurrentGameInfo().getAttackedAgent() != null) //initialize the first death
			previouslyAttacked = getCurrentGameInfo().getAttackedAgent();

		if(previouslyAttacked != getCurrentGameInfo().getAttackedAgent()){
			previouslyAttacked = getCurrentGameInfo().getAttackedAgent();
			noDeath = false;
		} else {
			noDeath = true;
			setTrustCoef(lastProtected, 100);
		}
		return;
	}

	/**
	 * Talk function
	 */
	public String talk() {
		return villagerTalk();
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
					getComingoutMap().put(talker, content.getRole()); 
					if(talk.getText().contains("MEDIUM")||talk.getText().contains("SEER"))
					changeTrustCoef(talker, 15);
					if(talk.getText().contains("BODYGUARD"))
					changeTrustCoef(talker, -40);
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
					if (content.getTarget().equals(getMe())){
						changeTrustCoef(talker, -10);
						break;
					}
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

	public Agent divine() {
		throw new UnsupportedOperationException();
	}

	public Agent attack() {
		throw new UnsupportedOperationException();
	}

	/**
	 * Guard function, called at the beginning of the night
	 * Protect the agent with the highest trust coef or continu to protect the agent that was attacked but not killed
	 */
	public Agent guard() {
		List<Agent> candidates = new ArrayList<>();

		if(lastProtected != null && getCurrentGameInfo().getAliveAgentList().contains(lastProtected) && noDeath)
			return lastProtected;

		for (Agent agent : getCurrentGameInfo().getAgentList()) {
			if (getComingoutMap().containsKey(agent) && !agent.equals(lastProtected) && getCurrentGameInfo().getAliveAgentList().contains(agent))
				candidates.add(agent);
		}

		if(!candidates.isEmpty()){
			lastProtected = randomSelect(candidates);
			return lastProtected;
		}

		for (Agent agent : getCurrentGameInfo().getAgentList()) {
			if (getTrustCoef(agent) <= 75 && !agent.equals(lastProtected) && getCurrentGameInfo().getAliveAgentList().contains(agent))
				candidates.add(agent);
		}
		
		lastProtected = randomSelect(candidates);
		return lastProtected;
	}

	public String whisper() {
		throw new UnsupportedOperationException();
	}

	public void finish(){

	}
}
