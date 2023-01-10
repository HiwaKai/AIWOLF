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

/**
 *
 * @author Victor
 */
public class RoleAssignPlayer extends org.aiwolf.sample.lib.AbstractRoleAssignPlayer {

    public RoleAssignPlayer(){
        setVillagerPlayer(new Villager());
		setBodyguardPlayer(new Bodyguard());
		setMediumPlayer(new Medium());
		setSeerPlayer(new Seer());
		setPossessedPlayer(new Possessed());
		setWerewolfPlayer(new Werewolf());
    }

    @Override
    public String getName(){
        return "baguette";
    }
}
