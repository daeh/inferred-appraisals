///////////// head /////////////

var debug = false,
	revoke = false;

var shuffledEmotionLabels = shuffleArray(["Amusement", "Annoyance", "Confusion", "Contempt", "Devastation", "Disappointment", "Disgust", "Embarrassment", "Envy", "Excitement", "Fury", "Gratitude", "Guilt", "Joy", "Pride", "Regret", "Relief", "Respect", "Surprise", "Sympathy"]);
randomizeEmotions(shuffledEmotionLabels);

/// initialize sliders ///
$('.not-clicked').mousedown(function() {
	$(this).removeClass('not-clicked').addClass('slider-input');
	$(this).closest('.slider').children('.slider-label-not-clicked').removeClass('slider-label-not-clicked').addClass('slider-label');
});

/// initialize rollover effects of emotion labels ///
$('.emotionRatingRow').hover(
	function() {
		$('#respRightEDisplayText').html($(this).children('td').children('div.slider').children('.isCenter').text());
		$(this).closest("tr").children("td").children("span.eFloor").html('not&nbsp;any');
		$(this).closest("tr").children("td").children("span.eFloor").closest("td").click(function() {
			$(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val('0');
			$(this).closest("tr").children("td").children("div.slider").children('.slider-label-not-clicked').removeClass('slider-label-not-clicked').addClass('slider-label');
			$(this).closest("tr").children("td").children("div.slider").children('.not-clicked').removeClass('not-clicked').addClass('slider-input');
			$("#barChart").height(Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val()));
		});
		$(this).closest("tr").children("td").children("span.eCeiling").html('immense');
		$(this).closest("tr").children("td").children("span.eCeiling").closest("td").click(function() {
			$(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val('100');
			$(this).closest("tr").children("td").children("div.slider").children('.slider-label-not-clicked').removeClass('slider-label-not-clicked').addClass('slider-label');
			$(this).closest("tr").children("td").children("div.slider").children('.not-clicked').removeClass('not-clicked').addClass('slider-input');
			$("#barChart").height(Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val()));
		});
		$("#barChart").height(Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val()));
	},
	function() {
		$('#respRightEDisplayText').html('&nbsp;');
		$(this).closest("tr").children("td").children("span.eFloor").html('&nbsp;');
		$(this).closest("tr").children("td").children("span.eCeiling").html('&nbsp;');
		$("#barChart").height(0);
	}
);

// $('.emotionRatingRow').mouseup(function() {$('#TEMP').html( $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val() ) } );
// $('.emotionRatingRow').mousemove(function() { $('#TEMP').html($(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val()) });
$('.emotionRatingRow').mouseup(function() {
	$("#barChart").height(
		Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val())
	);
});
$('.emotionRatingRow').mousemove(function() {
	$("#barChart").height(
		Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val())
	);
});


///////////// Window /////////////


/// MTURK Initalization ///

var serverRoot = ""; // requires terminal slash
var stimPath = "../stimuli/"; // requires terminal slash

/// UX Control ///

function interferenceHandler(e) {
	e.stopPropagation();
	e.preventDefault();
}

function checkPreview() {
	if (!!turk.previewMode) {
		alert("Please accept this HIT to see more questions.");
		return false;
	}
	return true;
}


/// User Input Control ///

// Radio Buttons //
function getRadioCheckedValue(formNum, radio_name) {
	var oRadio = document.forms[formNum].elements[radio_name];
	for (var i = 0; i < oRadio.length; i++) {
		if (oRadio[i].checked) {
			return oRadio[i].value;
		}
	}
	return '';
}

function getRadioResponse(radioName) {
	var radios = document.getElementsByName(radioName);
	for (var i = 0; i < radios.length; i++) {
		if (radios[i].checked == true) {
			return radios[i].value;
		}
	}
	return '';
}

function ResetRadios(radioName) {
	var radios = document.getElementsByName(radioName);
	for (var i = 0; i < radios.length; i++) {
		radios[i].checked = false;
	}
}

function array_mean(ain) {
	let asum = 0;
	for (let i = 0; i < ain.length; i++) {
		asum += parseInt(ain[i]);
	}
	return asum/ain.length;
}

function array_range(ain) {
	let anum = [];
	for (let i = 0; i < ain.length; i++) {
		anum.push(parseInt(ain[i]));
	}
	return Math.max(...anum) - Math.min(...anum);
}

function validate_response_filter(trial) {
	let resp_valid = true;

	let neg_mean = array_mean([trial.e_devastation, trial.e_disappointment, trial.e_fury, trial.e_annoyance]);
	let pos_mean = array_mean([trial.e_relief, trial.e_excitement, trial.e_joy]);
	let e_range = array_range([trial.e_amusement, trial.e_annoyance, trial.e_confusion, trial.e_contempt, trial.e_devastation, trial.e_disappointment, trial.e_disgust, trial.e_embarrassment, trial.e_envy, trial.e_excitement, trial.e_fury, trial.e_gratitude, trial.e_guilt, trial.e_joy, trial.e_pride, trial.e_regret, trial.e_relief, trial.e_respect, trial.e_surprise, trial.e_sympathy]);

	if (trial.decisionOther === 'Split') {
		if (neg_mean >= pos_mean && trial.pot > 2000) {
			resp_valid = false;
			if (debug) {
				console.log('e_neg_mean >= e_pos_mean fail');
			}
		}
		if (neg_mean/48 > 0.3 && trial.pot > 2000) {
			resp_valid = false;
			if (debug) {
				console.log('e_neg_mean fail');
			}
		}
	}

	if (trial.decisionOther === 'Stole') {
		if (pos_mean >= neg_mean) {
			resp_valid = false;
			if (debug) {
				console.log('e_pos_mean >= e_neg_mean fail');
			}
		}
		if (pos_mean/48 > 0.3) {
			resp_valid = false;
			if (debug) {
				console.log('e_pos_mean fail');
			}
		}
	}

	if (e_range/48 < 0.2) {
		resp_valid = false;
		if (debug) { console.log('e_range fail'); }
	}

	if (debug) {
		console.log([trial.e_devastation, trial.e_disappointment, trial.e_fury, trial.e_annoyance]);
		console.log('neg_mean ', neg_mean / 48, '    ;  pos_mean ', pos_mean / 48, '    ; range: ', e_range / 48);
		console.log('111 ', trial.decisionOther === 'Split' & neg_mean / 48 > 0.3);
		console.log('222 ', trial.decisionOther === 'Stole' & pos_mean / 48 > 0.3);
		console.log('333 ', e_range / 48 < 0.2);
	}

	return resp_valid;
}

function ValidateRadios(radioNameList) {
	if (debug === true) {return true;}
	for (var j = 0; j < radioNameList.length; j++) {
		var pass = false;
		var radios = document.getElementsByName(radioNameList[j]);
		for (var i = 0; i < radios.length; i++) {
			if (radios[i].checked === true) {
				pass = true;
				i = radios.length;
			}
		}
		if (pass === false) {
			alert("Please provide an answer to every question.");
			return pass;
		}
	}
	return pass;
}

function ValidateDemoRadios(radioName, expectedAnswer) {
	var pass = false;
	if (getRadioResponse(radioName) == expectedAnswer) {pass = true;}
	if (pass === false) {
		alert("Please check your response to what this player expects the other player to choose");
		return pass;
	}
	return pass;
}

function validateResponses(responses, expected) {
	var pass = true;
	if (responses.length != expected.length) {
		pass = false;
		// console.log('valid size missmatch ' + responses.length + '  ' + expected.length)
	} else {
		for (var i=0; i<expected.length; i++) {
			if (expected[i] != 'bypass') {
				if (responses[i] != expected[i]) {
					pass = false;
					// console.log(responses[i] + ' vs ' + expected[i]);
				}
			}
		}
	}
	return pass;
}


// Ranges //
function ValidateRanges() {
	var pass = true;
	var unanswered = document.querySelector("#responsesTable").querySelectorAll(".slider-label-not-clicked");
	if (unanswered.length > 0 && !debug) {
		pass = false;
		alert("Please provide an answer to every question.");
		//alert("Please provide an answer to all emotions. If you think that a person is not experiencing a given emotion, rate that emotion as --not at all-- by moving the sliding marker all the way to the left.");
	}
	return pass;
}

function validateDemoRanges() {
	var pass = true;
	var unanswered = document.querySelector("#demoTable").querySelectorAll(".slider-label-not-clicked");
	if (!debug) {
		if (unanswered.length > 0) {
			pass = false;
			alert("Please provide an answer to all emotions.");
		} else if (demo1.value != 0) {
			pass = false;
			alert("The sliding range of --excitement-- is not set to the minimum possible value. Please click the --not any-- text to the left of the grey bar.");
		} else if (demo2.value < 10 || demo2.value > 38) {
			pass = false;
			alert("The sliding range of --contempt-- is not set near the mid-point of the grey bar. Please move the marker towards the middle.");
		} else if (demo3.value != 48) {
			pass = false;
			alert("The sliding range of --fury-- is not set all the way to the maximum possible value. Please click and drag the marker all the way to the right of the grey bar.");
		}
	}
	return pass;
}

function ResetRanges() {
	var ranges = $('input[type="range"]');
	for (var i = 0; i < ranges.length; i++) {
		ranges[i].value = "0";
	}
	ranges.removeClass('slider-input').addClass('not-clicked');
	ranges.closest('.slider').children('label').removeClass("slider-label").addClass("slider-label-not-clicked");
}

// Textareas //
function ValidateFieldEquivalence(testField, targetString) {
	return ValidateTextEquivalence(testField.value, targetString);
}

function ValidateTextEquivalence(test, target) {
	var valid = true;
	var parsedTarget = parseWords(target);
	var parsedField = parseWords(test);
	if (!(parsedTarget.length === parsedField.length)) {
		valid = false;
	} else {
		for (var i = 0; i < parsedTarget.length; i++) {
			if (!(parsedTarget[i].toUpperCase() === parsedField[i].toUpperCase())) {
				valid = false;
			}

		}
	}
	return valid;
}

function ValidateText(field, min, max) {
	var valid = true;

	if (field.value === "") {
		alert("Please provide an answer.");
		valid = false;
	} else {
		var values = parseWords(field.value);
		if (values.length > max || values.length < min) {
			// invalid word number
			return false;
		}
	}
	return valid;
}

function parseWords(string) {
	// !variable will be true for any falsy value: '', 0, false, null, undefined. null == undefined is true, but null === undefined is false. Thus string == null, will catch both undefined and null.
	// (typeof string === 'undefined' || !string)
	var values = "";
	if (!!string) {
		values = string.replace(/\n/g, " ").split(' ').filter(function(v) {
			return v !== '';
		});
	}
	return values;
}


/// Generic Functions ///

function genIntRange(min, max) {
	var range = [];
	for (var i = min; i <= max; i++) {
		range.push(i);
	}
	return range;
}

/**
 * Randomize array element order in-place.
 * Using Durstenfeld shuffle algorithm.
 */
function shuffleArray(array) {
	for (var i = array.length - 1; i > 0; i--) {
		var j = Math.floor(Math.random() * (i + 1));
		var temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
	return array;
}

/* 
random(a,b)
Returns random number between a and b, inclusive
*/
function random(a, b) {
	if (typeof b == "undefined") {
		a = a || 2;
		return Math.floor(Math.random() * a);
	} else {
		return Math.floor(Math.random() * (b - a + 1)) + a;
	}
}

/* 
Array.prototype.random
Randomly shuffles elements in an array. Useful for condition randomization.

E.G.
var choices = ["rock", "paper", "scissors"];
var computer = { "play":choices.random() };

console.log("The computer picked " + computer["play"]+ ".");
*/
Array.prototype.random = function() {
	return this[random(this.length)];
};

var trial_timer = 0;
function timeEvent(t,i) {
	if (i === 'initialize') {
		t = new Date().getTime();
		return t;
	} else if (i === 'lap_reset') {
		var t1 = new Date().getTime() - t;
		t = new Date().getTime();
		return t1/1000;
	} else if (i === 'mark_diff') {
		var t1 = new Date().getTime() - t;
		return t1/1000;
	}
}

function detect_browser() {
	var output = [];

	// Opera 8.0+
	var isOpera = (!!window.opr && !!opr.addons) || !!window.opera || navigator.userAgent.indexOf(' OPR/') >= 0;
	if (isOpera) {output.push('Opera')}

	// Firefox 1.0+
	var isFirefox = typeof InstallTrigger !== 'undefined';
	if (isFirefox) {output.push('Firefox')}

	// Safari 3.0+ "[object HTMLElementConstructor]"
	var isSafari = /constructor/i.test(window.HTMLElement) || (function (p) { return p.toString() === "[object SafariRemoteNotification]"; })(!window['safari'] || (typeof safari !== 'undefined' && safari.pushNotification));
	if (isSafari) {output.push('Safari')}

	// Internet Explorer 6-11
	var isIE = /*@cc_on!@*/false || !!document.documentMode;
	if (isIE) {output.push('IE')}

	// Edge 20+
	var isEdge = !isIE && !!window.StyleMedia;
	if (isEdge) {output.push('Edge')}

	// Chrome 1 - 71
	var isChrome = !!window.chrome && (!!window.chrome.webstore || !!window.chrome.runtime);
	if (isChrome) {output.push('Chrome')}

	// Blink engine detection
	var isBlink = (isChrome || isOpera) && !!window.CSS;
	if (isBlink) {output.push('Blink')}

	return output;
}

/// Window Control ///

// Avoids progress loss that happens via back button, closing tab, refreshing window etc
function avoidWindowUnload() {
	window.addEventListener('beforeunload', function(e) {
		if (maintaskParam.numComplete >= 1 && maintaskParam.numComplete < maintaskParam.numTrials) {
			// Cancel the event
			e.preventDefault();
			// Chrome requires returnValue to be set
			e.returnValue = '';
		}
	});
}
avoidWindowUnload();

function showSlide(id) {
	$(".slide").hide();
	$("#" + id).show();
	$("#responseDivCues0").show();
}

function presentStim() {
	if (nPresentations === 0) {
		smallVideo.src = document.getElementById("videoStim").src;
		setTimeout(smallVideo.pause(), 300);
		setTimeout(function() { smallVideo.currentTime = 5.04; }, 1000);

		document.addEventListener("clicks", interferenceHandler, true);
		document.addEventListener("contextmenu", interferenceHandler, true);
		$('#interactionMask').show();
		$('#videoStimPackageDiv').css('opacity', '0');
		$("#videoStimPackageDiv").show();
		disablePlayButton();
		$("#playButtonContainer").hide();
		playVid();
	}
}


function presentResponses() {
	showSlide("slideResponse");
	// getResponseDivSizes();

	$("#contextTableFrontDiv").html('&nbsp;');

	if (maintaskParam.numComplete <= 1) {
		setResponseTableSizes();
	}
	$(".responseBarChartBackground").height(getResponseDivSizes());

	// $('#outer_responseCollection').show();

	// $("#responseTableFrame").height($('#responseTableFrameID').outerHeight());

	// $('#outer_stimPresentation').hide();
	// $('#outer_stimPresentation').addClass('element_purge'); // totally overkill could mess with alignment


	// start response timer
	trial_timer = timeEvent(0,'initialize');
	console.log('starting response timer  ', trial_timer); //TEMP

	window.scrollTo(0,0);
}

function setResponseTableSizes() {
	// console.log('$("#responseTableFrameID").height()', $("#responseTableFrameID").height());
	// console.log('$("#responseCuesTableFrameID").height()', $("#responseCuesTableFrameID").height());
	// console.log('',);

	// var minTableHeight = Math.max($("#responseTableFrameID").height(), $("#responseCuesTableFrameID").height());

	// $("#responseTableFrameID").height(minTableHeight)
	// $("#responseCuesTableFrameID").height(minTableHeight);

	var tableHeight = getResponseDivSizes(); // (30 is cell padding + border)*2  //This works in chrome and safari, but makes the frame 30 too big in Firefox DEBUG

	// console.log('tableHeight', tableHeight);

	$("#responseTableFrameID").height(tableHeight);
	$("#responseCuesTableFrameID").height(tableHeight);

	// return minTableHeight;
}

function getResponseDivSizes() {
	return ($('#RespSet').outerHeight());
}

/// Stim Control ///

// var smallVideo = document.getElementById("videoStim_small");

// function setSmallVid()
// {
//     var promise = longfunctionfirst().then(shortfunctionsecond);
// }
// function longfunctionfirst()
// {
//     var d = new $.Deferred();
//     setTimeout(function() {
// 
// d.resolve();
//     },1);
//     return d.promise();
// }
// function shortfunctionsecond()
// {
//     var d = new $.Deferred();
//     setTimeout(function() {
//     	smallVideo.play();
//     	smallVideo.pause();
//     	smallVideo.currentTime = 5;
//     	d.resolve();
//     },10);
//     return d.promise();
// }

var thisVideo = document.getElementById("videoStim");
var nPresentations = 0;

function stimHandler() {
	document.getElementById('videoStim').removeEventListener('ended', stimHandler, false);
	setTimeout(function() {
		$('#videoStimPackageDiv').css('opacity', '0');
		if (nPresentations < 2) {
			setTimeout(playVid, 1000);
		} else {
			$("playButtonContainer").hide();
			document.removeEventListener("clicks", interferenceHandler, true);
			document.removeEventListener("contextmenu", interferenceHandler, true);
			$('#interactionMask').hide();

			presentResponses();
			preloadStim(maintaskParam.numComplete); // load next movie
		}
	}, 1000);
}

function playVid() {
	nPresentations++;
	thisVideo.pause();
	thisVideo.currentTime = 0;

	setTimeout(function() {
		$('#videoStimPackageDiv').css('opacity', '1');
		setTimeout(function() {
			thisVideo.play();
			document.getElementById('videoStim').addEventListener('ended', stimHandler, false);
		}, 1000);
	}, 1000);
}

/*
// FETCH calls not supported by legacy browsers (or currently iOS). Using XML instead for the time being.
// load next video
function preloadStim(stimNum) {
	// var tempurl = serverRoot + stimPath + "dynamics/" + maintaskParam.allConditionStim[maintaskParam.shuffledOrder[ stimNum ]].stimulus + "t.mp4";
	fetch(serverRoot + stimPath + "dynamics/" + maintaskParam.allConditionStim[maintaskParam.shuffledOrder[stimNum]].stimulus + "t.mp4").then(function(response) {
		return response.blob();
	}).then(function(data) {
		document.getElementById("videoStim").src = URL.createObjectURL(data);
		enablePlayButton();
		console.log("Current load: stimNum    ", stimNum, "maintaskParam.shuffledOrder[ stimNum ]                  ", maintaskParam.shuffledOrder[stimNum], "stimID", maintaskParam.allConditionStim[maintaskParam.shuffledOrder[stimNum]].stimulus, "( maintaskParam.numComplete", maintaskParam.numComplete, ")");
		// console.log("maintaskParam.numComplete", maintaskParam.numComplete, "maintaskParam.shuffledOrder[ maintaskParam.numComplete ]", maintaskParam.shuffledOrder[ maintaskParam.numComplete ], "stimID", maintaskParam.allConditionStim[maintaskParam.shuffledOrder[ maintaskParam.numComplete ]].stimulus);
	}).catch(function() {
		console.log("Booo");
	});
}
*/

function preloadStim(stimNum) {
	var req = new XMLHttpRequest();
	var stimURL = serverRoot + stimPath + "dynamics/" + maintaskParam.allConditionStim[maintaskParam.shuffledOrder[stimNum]].stimulus + "t.mp4";
	req.open('GET', stimURL, true);
	req.responseType = 'blob';

	req.onload = function() {
		// Onload is triggered even on 404
		// so we need to check the status code
		if (this.status === 200) {
			var videoBlob = this.response;
			document.getElementById("videoStim").src = URL.createObjectURL(videoBlob); // IE10+
			enablePlayButton();
			console.log("Current load: stimNum    ", stimNum, "maintaskParam.shuffledOrder[ stimNum ]                  ", maintaskParam.shuffledOrder[stimNum], "stimID", maintaskParam.allConditionStim[maintaskParam.shuffledOrder[stimNum]].stimulus, "( maintaskParam.numComplete", maintaskParam.numComplete, ")");
		}
	};
	req.onerror = function() {
		console.log("Booo");
	};
	req.send();
}


// Play button control //
function disablePlayButton() {
	$('#playButton').prop('onclick', null).off('click');
	$('#playButton').removeClass('play-button').addClass('play-button-inactive');
	$("#loadingTextLeft").html('VIDEO&nbsp;');
	$("#loadingTextRight").html('&nbsp;LOADING');
}

function enablePlayButton() {
	$('#playButton').click(function() {
		presentStim();
		$("#responseDivCues0").hide();
	});
	$('#playButton').removeClass('play-button-inactive').addClass('play-button');
	$("#loadingTextLeft").html('&nbsp;');
	$("#loadingTextRight").html('&nbsp;');
	nPresentations = 0;
}

function numberWithCommas(number) {
	return number.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ","); // insert commas in thousands places in numbers with less than three decimals
	// return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ","); // insert commas in thousands places in numbers with less than three decimals
}


var mmtofurkeyGravy = {
	// updateRandCondTable: function(rawData_floorLevel, turk) {

	// rawData_floorLevel.randC.....
	// },

	doNothing: function(rawData_floorLevel, turk) {
		// http://railsrescue.com/blog/2015-05-28-step-by-step-setup-to-send-form-data-to-google-sheets/
		try {
			console.log("happy eating");
		} catch (e) { console.log("tofurkeyGravy Error:", e); }
	}
};

var trainingVideo = {
	preloadStim: function() {
		var stimURL = serverRoot + stimPath + "dynamics/" + "258_c_ed_vbr2.mp4";
		// var stimURL = serverRoot + stimPath + "dynamics/" + "258_context1_vbr1_H265.mp4";
		if (!debug) {
			var req = new XMLHttpRequest();
			req.open('GET', stimURL, true);
			req.responseType = 'blob';
			req.onload = function() {
				// Onload is triggered even on 404
				// so we need to check the status code
				if (this.status === 200) {
					var videoBlob = this.response;
					document.getElementById("videoStim_training").src = URL.createObjectURL(videoBlob); // IE10+
					// document.getElementById("videoStim_training").src = stimURL;
					console.log("Current load: training complete    ", document.getElementById("videoStim_training").src);
					// $('#playButton_training').click(function() {
					// 	presentStim();
					// });
					$("#loadingTextLeft_training").html('&nbsp;');
					$("#loadingTextRight_training").html('&nbsp;');
					$("#videoLoadingDiv").hide();
					// $("#videoStimPackageDiv_training").show()
					// console.log("Current load: stimNum    ", stimNum);

					document.getElementById('videoStim_training').addEventListener('ended', enableAdvance, false);
				}
			};
			req.onerror = function() {
				console.log("Booo");
			};
			req.onprogress = function(oEvent) {
				if (oEvent.lengthComputable) {
					var percentComplete = oEvent.loaded/oEvent.total;
					document.getElementById("trainingVideoProgress").textContent = Math.floor(percentComplete*100) + '%';
				}
			}
			req.send();
		} else {
			document.getElementById("videoStim_training").src = stimURL;
			$("#loadingTextLeft_training").html('&nbsp;');
			$("#loadingTextRight_training").html('&nbsp;');
			$("#videoLoadingDiv").hide();
			enableAdvance();
			console.log("bypassing training blob");
		}
	}
};

var preload_static_images = {
	preload: function() {
		this.images = [
				// {id: "exampleRange1", style: "width: 457; height: 32px;", alt: "Player Values", src: "images/ExampleRange.png"},
				// {id: "exampleRange2", style: "width: 900px; height: 162px;", alt: "Player Values", src: "images/exampleRange2.png"},
			];

		for (var i = 0; i < this.images.length; i++) {
			this.images[i].imgObj = new Image();
			this.images[i].imgObj.src = this.images[i].src;
			this.images[i].imgObj.style = this.images[i].style;
			this.images[i].imgObj.alt = this.images[i].alt;
			document.getElementById( this.images[i].id ).appendChild(this.images[i].imgObj);
			console.log('preloaded static image :', this.images[i].id);
		}
	}
};

var preload_named_images = {
	preload: function() {
		this.images = {
			CC: {src: "images/CC.png"},
			CD: {src: "images/CD.png"},
			DC: {src: "images/DC.png"},
			DD: {src: "images/DD.png"},
		};

		Object.values(this.images).forEach(value => {
			value.imgObj = new Image();
			value.imgObj.src = value.src;
			value.imgObj.style = "width: 648px; height: 115px;";
			value.imgObj.alt = "Outcome Context";
			console.log('preloaded named image :', value.src);
		})
	},

	replace: function(target_id, imgObj_name) {
		var target = document.getElementById(target_id);
		console.log(target);
		if (target.childElementCount === 0) {
			target.appendChild(this.images[imgObj_name].imgObj);
		} else {
			target.replaceChild(this.images[imgObj_name].imgObj, target.firstChild);
		}
	}
};

var v0_response = {

	v0_response: '',
	expectedResponse: '',

	validate: function(expectedResponse) {
		var radios = document.getElementsByName('v0');
		var radiosValue = false;

		for (var i = 0; i < radios.length; i++) {
			if (radios[i].checked == true) {
				radiosValue = true;
			}
		}
		if (expectedResponse === 'bypass') {
			return true;
		} else {
			if (!radiosValue) {
				alert("Please watch the video and answer the question");
				return false;
			} else {
				this.v0_response = getRadioResponse("v0");
				this.expectedResponse = expectedResponse;
				return true;
			}
		}
	}

};

function enableAdvance() {

	// load the HIT
	$('#training_advance_button').removeClass('advance-button-inactive').addClass('advance-button');
	if (!debug) {
		document.getElementById('training_advance_button').onclick = function() {
			if(!!v0_response.validate('7510')){this.blur(); loadHIT(4); window.scrollTo(0,0);}
		};
	} else {
		document.getElementById('training_advance_button').onclick = function() {
			if(!!v0_response.validate('bypass')){this.blur(); loadHIT(4); window.scrollTo(0,0);}
		};
	}

	// Preload other page elements
	preload_static_images.preload();
	preload_named_images.preload();
}

///WIP
function capture_hit_info() {
	var assignment_id_field = document.getElementById('myelementLink');
	var paramstr = window.location.search.substring(1);
	var parampairs = paramstr.split("&");
	var mturkworkerID = "";
	for (i in parampairs) {
		var pair = parampairs[i].split("=");
		if (pair[0] == "workerId")
			mturkworkerID = pair[1];
	}
	return mturkworkerID;
}

/// Experiment ///

function SetMaintaskParam(selectedTrials) {

	this.allConditions = [returnStimuli()];

	/* Experimental Variables */

	// Keep track of how many trials have been completed
	this.numComplete = 0;

	// Number of conditions in experiment
	this.numConditions = this.allConditions.length;

	// Randomly select a condition number for this particular participant
	this.chooseCondition = 1; // random(0, numConditions-1);

	// Based on condition number, choose set of input (trials)
	this.allConditionStim = this.allConditions[this.chooseCondition - 1]; // all stim in condition unshuffled

	// Produce random order in which the trials will occur
	// this.shuffledOrder = shuffleArray(genIntRange(0, this.allConditionStim.length - 1));
	this.shuffledOrder = shuffleArray(selectedTrials);

	// Pull the random subet
	this.subsetStimOrdered = [];
	for (var i = 0; i < this.shuffledOrder.length; i++) {
		this.subsetStimOrdered.push(this.allConditionStim[this.shuffledOrder[i]]);
	}

	// this.randPotSize = shuffleArray([10300.00, 19100.00, 33560.00, 48650.00, 67380.00, 84700.00, 140800.00, 162030.00]);
	//
	// if (this.randPotSize.length != this.subsetStimOrdered.length) {console.log('WARNING -- size mismatch', this.randPotSize.length, " vs ", this.subsetStimOrdered.length);}
	// for (var j = 0; j < this.randPotSize.length; j++) {
	// 	this.subsetStimOrdered[j]["pot"] = this.randPotSize[j];
	// }

	// add practice first stim
	var demo_stim = { "stimulus": "244_2", "pronoun":"this person", "desc": "", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1090.00, "index": -1 };
	this.subsetStimOrdered.unshift(demo_stim);

	this.storeDataInSitu = false;

	// Number of trials
	this.numTrials = this.subsetStimOrdered.length; //not necessarily this.allConditionStim.length;
}

function randomizeEmotions(emotionLabels) {
	var emoTable = document.getElementById('responsesTable');
	var emoTableBod = document.getElementById('responsesTable').getElementsByTagName('tbody')[0];
	for (var i = 0; i < emotionLabels.length; i++) {
		var row = 	'<tr class="emotionRatingRow">' +
					'<td width="0" align="left" valign="middle" nowrap class="outsideEmotionLabels restText"><strong>' + emotionLabels[i] + '&nbsp;</strong></td>' +
					'<td height="25" style="text-align: right;"><span class="eFloor"> &nbsp;</span></td>' +
					'<td align="center" valign="middle" width="301">' +
					'<div class="slider">' +
					'<label id="slider-label-' + emotionLabels[i].toLowerCase() + '" class="slider-label-not-clicked isCenter">' + emotionLabels[i] + '</label>' +
					'<input id="e_' + emotionLabels[i].toLowerCase() + '" class="not-clicked" type="range" min="0" value="-1" max="48">' +
					'</div>' +
					'</td>' +
					'<td height="25" style="text-align: left;"><span class="eCeiling"> &nbsp;</span></td>' +
					'</tr>';
		$('#responsesTable').find('tbody').append(row);
		// emoTableBod.appendChild(row);
	}
}

var phpParam = {
	baseURL: 'https://daeda.scripts.mit.edu/serveCondition/serveCondition.php?callback=?', 
	condfname: "servedConditions_exp11a.csv"};
var maintask = [];
var maintaskParam = [];
var done_preloading = false;

function loadHIT(nextSlide) {
	// determine if HIT is live
	var assignmentId_local = turk.assignmentId,
		turkSubmitTo_local = turk.turkSubmitTo;

	// If there's no turk info
	phpParam.writestatus = "TRUE";
	if (!assignmentId_local || !turkSubmitTo_local) {
		console.log("Dead Turkey: Not writing to conditions file.");
		phpParam.writestatus = "FALSE";
	} else {
		console.log("Live Turkey!");
		phpParam.writestatus = "TRUE";
		debug = false;
	}

	var requestService = 'writestatus=' + phpParam.writestatus + '&condComplete=' + 'REQUEST' + '&condfname=' + phpParam.condfname;

	showSlide("loadingHIT");

	$.getJSON(phpParam.baseURL, requestService, function(res) {

		console.log("Served Condition:", res.condNum);

		showSlide(nextSlide);

		var conditions = returnConditions();
		var defaultCondNum = shuffleArray(genIntRange(0,conditions.length-1))[0]; // if PHP runs out of options

		var condNum = (parseInt(res.condNum) >= 0 && parseInt(res.condNum) <= conditions.length - 1) ? parseInt(res.condNum) : defaultCondNum * -1;
		var selectedTrials = conditions[Math.abs(condNum)];

		if (!!debug) {
			console.log('Delivering Condition:  ', condNum);
		}

		maintaskParam = new SetMaintaskParam(selectedTrials);

		// Preloading images
		if (!done_preloading) {
			var playerImages = [];
			for (var i = 0; i < maintaskParam.subsetStimOrdered.length; i++) {
				playerFace = new Image();
				var image_name = maintaskParam.subsetStimOrdered[i].stimulus;
				playerFace.src = serverRoot + stimPath + "statics/" + image_name + ".png"; // `./stimuli/statics/${image_name}.png`;
				playerFace.style = 'width:250px; height:250px;';
				playerFace.alt = 'Player';
				console.log("Trial[", i, "]::  ", playerFace);
				playerImages.push(playerFace)
			}
			done_preloading = true;
		}

		// Updates the progress bar
		$("#trial-num").html(maintaskParam.numComplete);
		$("#total-num").html(maintaskParam.numTrials);

		maintask = {

			respTimer: new Array(maintaskParam.numTrials),
			stimulusArray: new Array(maintaskParam.numTrials),
			stimDescArray: new Array(maintaskParam.numTrials),

			e_amusement: new Array(maintaskParam.numTrials),
			e_annoyance: new Array(maintaskParam.numTrials),
			e_confusion: new Array(maintaskParam.numTrials),
			e_contempt: new Array(maintaskParam.numTrials),
			e_devastation: new Array(maintaskParam.numTrials),
			e_disappointment: new Array(maintaskParam.numTrials),
			e_disgust: new Array(maintaskParam.numTrials),
			e_embarrassment: new Array(maintaskParam.numTrials),
			e_envy: new Array(maintaskParam.numTrials),
			e_excitement: new Array(maintaskParam.numTrials),
			e_fury: new Array(maintaskParam.numTrials),
			e_gratitude: new Array(maintaskParam.numTrials),
			e_guilt: new Array(maintaskParam.numTrials),
			e_joy: new Array(maintaskParam.numTrials),
			e_pride: new Array(maintaskParam.numTrials),
			e_regret: new Array(maintaskParam.numTrials),
			e_relief: new Array(maintaskParam.numTrials),
			e_respect: new Array(maintaskParam.numTrials),
			e_surprise: new Array(maintaskParam.numTrials),
			e_sympathy: new Array(maintaskParam.numTrials),

			randCondNum: new Array(1),
			
			iwould_large: new Array(0),
			iwould_small: new Array(0),
			iexpectOther_large: new Array(0),
			iexpectOther_small: new Array(0),

			validationRadioExpectedResp: new Array(0),
			validationRadio: new Array(0),
			dem_gender: [],
			dem_language: [],
			val_recognized: [],
			val_familiar: [],
			val_feedback: [],

			emotionOrder: shuffledEmotionLabels,

			data: [],
			dataInSitu: [],

			total_time: timeEvent(0,'initialize'),
			visible_area: [document.documentElement.clientWidth, document.documentElement.clientHeight],
			browser: detect_browser(),
			browser_version: navigator.userAgent,

			validateRadioResponse: function(fieldname, expectedResponse, nextslide) {
				this.validationRadioExpectedResp.push(expectedResponse);
				this.validationRadio.push(getRadioResponse(fieldname));
				showSlide(nextslide);
			},

			finalForcedQuestions: function(fieldname1, fieldname2, nextslide) {
				if (getRadioResponse(fieldname1) === '' || getRadioResponse(fieldname2) === '') {
					alert("Please answer the question");
					return false;
				} else {
					this[fieldname1].push(getRadioResponse(fieldname1));
					this[fieldname2].push(getRadioResponse(fieldname2));
				}
				showSlide(nextslide);
			},

			end: function() {
				// stop experiment timer
				this.total_time = timeEvent(this.total_time,'mark_diff');

				var subjectValid = validateResponses(this.validationRadio, ['7510','disdain','jealousy','AF25HAS','steal','bypass','bypass']);
				// if (!!subjectValid) {
				// SEND DATA TO TURK
				// }

				this.dem_gender = getRadioResponse("d1");
				this.dem_language = $('textarea[name="dem_language"]').val();
				this.val_recognized = $('textarea[name="val_recognized"]').val();
				this.val_familiar = $('textarea[name="val_familiar"]').val();
				this.val_feedback = $('textarea[name="val_feedback"]').val();

				// SEND DATA TO TURK
				setTimeout(function() {
					turk.submit(maintask, true, mmtofurkeyGravy);
					setTimeout(function() { showSlide("exit"); }, 1000);
				}, 1000);

				// DEBUG PUT THIS IN MMTOFURKYGRAVY AND ADD STRING PARAM TO MAINTASK VARIABLES
				console.log("attempting to return condition");
				var returnServe = 'writestatus=' + phpParam.writestatus + '&condComplete=' + maintask.randCondNum.toString() + '&subjValid=' + subjectValid.toString().toUpperCase() + '&condfname=' + phpParam.condfname;
				console.log('php: ' + returnServe);
				$.getJSON(phpParam.baseURL, returnServe, function(res) {
					console.log("Serve Returned!", res.condNum);
				});

				// Show the finish slide.
				$('#pietimerelement').pietimer({
					seconds: 2,
					color: 'rgb(76, 76, 76)',
					height: 200,
					width: 200
				});
				showSlide("finished");
				$('#pietimerelement').pietimer('start');
			},

			next: function() {
				// Show the experiment slide.
				// $("#videoStimPackageDiv").hide();
				// $("#playButtonContainer").show();

				// duplicate allConditionStim
				if (maintaskParam.numComplete === 0) { // if this is the first trial

					// add validation responses from training
					this.validationRadio.push(v0_response.v0_response);
					this.validationRadioExpectedResp.push(v0_response.expectedResponse);

					this.randCondNum = condNum;
					// disablePlayButton();
					// preloadStim(0); // load first video

					if (!!maintaskParam.storeDataInSitu) {
						for (var i = 0; i < maintaskParam.allConditionStim.length; i++) {
							var temp = maintaskParam.allConditionStim[i];
							temp.e_amusement = "";
							temp.e_annoyance = "";
							temp.e_confusion = "";
							temp.e_contempt = "";
							temp.e_devastation = "";
							temp.e_disappointment = "";
							temp.e_disgust = "";
							temp.e_embarrassment = "";
							temp.e_envy = "";
							temp.e_excitement = "";
							temp.e_fury = "";
							temp.e_gratitude = "";
							temp.e_guilt = "";
							temp.e_joy = "";
							temp.e_pride = "";
							temp.e_regret = "";
							temp.e_relief = "";
							temp.e_respect = "";
							temp.e_surprise = "";
							temp.e_sympathy = "";

							this.dataInSitu.push(temp);
							//// CLEAN this up a little?
						}
					}
				}

				// If this is not the first trial, record variables
				if (maintaskParam.numComplete > 0) { // If this isn't the first time .next() has been called

					// record last response time
					maintaskParam.trial.respTimer = timeEvent(trial_timer,'mark_diff');
					this.respTimer[maintaskParam.numComplete - 1] = maintaskParam.trial.respTimer;

					maintaskParam.trial.e_amusement = e_amusement.value;
					maintaskParam.trial.e_annoyance = e_annoyance.value;
					maintaskParam.trial.e_confusion = e_confusion.value;
					maintaskParam.trial.e_contempt = e_contempt.value;
					maintaskParam.trial.e_devastation = e_devastation.value;
					maintaskParam.trial.e_disappointment = e_disappointment.value;
					maintaskParam.trial.e_disgust = e_disgust.value;
					maintaskParam.trial.e_embarrassment = e_embarrassment.value;
					maintaskParam.trial.e_envy = e_envy.value;
					maintaskParam.trial.e_excitement = e_excitement.value;
					maintaskParam.trial.e_fury = e_fury.value;
					maintaskParam.trial.e_gratitude = e_gratitude.value;
					maintaskParam.trial.e_guilt = e_guilt.value;
					maintaskParam.trial.e_joy = e_joy.value;
					maintaskParam.trial.e_pride = e_pride.value;
					maintaskParam.trial.e_regret = e_regret.value;
					maintaskParam.trial.e_relief = e_relief.value;
					maintaskParam.trial.e_respect = e_respect.value;
					maintaskParam.trial.e_surprise = e_surprise.value;
					maintaskParam.trial.e_sympathy = e_sympathy.value;

					this.e_amusement[maintaskParam.numComplete - 1] = maintaskParam.trial.e_amusement;
					this.e_annoyance[maintaskParam.numComplete - 1] = maintaskParam.trial.e_annoyance;
					this.e_confusion[maintaskParam.numComplete - 1] = maintaskParam.trial.e_confusion;
					this.e_contempt[maintaskParam.numComplete - 1] = maintaskParam.trial.e_contempt;
					this.e_devastation[maintaskParam.numComplete - 1] = maintaskParam.trial.e_devastation;
					this.e_disappointment[maintaskParam.numComplete - 1] = maintaskParam.trial.e_disappointment;
					this.e_disgust[maintaskParam.numComplete - 1] = maintaskParam.trial.e_disgust;
					this.e_embarrassment[maintaskParam.numComplete - 1] = maintaskParam.trial.e_embarrassment;
					this.e_envy[maintaskParam.numComplete - 1] = maintaskParam.trial.e_envy;
					this.e_excitement[maintaskParam.numComplete - 1] = maintaskParam.trial.e_excitement;
					this.e_fury[maintaskParam.numComplete - 1] = maintaskParam.trial.e_fury;
					this.e_gratitude[maintaskParam.numComplete - 1] = maintaskParam.trial.e_gratitude;
					this.e_guilt[maintaskParam.numComplete - 1] = maintaskParam.trial.e_guilt;
					this.e_joy[maintaskParam.numComplete - 1] = maintaskParam.trial.e_joy;
					this.e_pride[maintaskParam.numComplete - 1] = maintaskParam.trial.e_pride;
					this.e_regret[maintaskParam.numComplete - 1] = maintaskParam.trial.e_regret;
					this.e_relief[maintaskParam.numComplete - 1] = maintaskParam.trial.e_relief;
					this.e_respect[maintaskParam.numComplete - 1] = maintaskParam.trial.e_respect;
					this.e_surprise[maintaskParam.numComplete - 1] = maintaskParam.trial.e_surprise;
					this.e_sympathy[maintaskParam.numComplete - 1] = maintaskParam.trial.e_sympathy;

					if (!!maintaskParam.storeDataInSitu) {
						maintaskParam.trialInSitu.e_amusement = maintaskParam.trial.e_amusement;
						maintaskParam.trialInSitu.e_annoyance = maintaskParam.trial.e_annoyance;
						maintaskParam.trialInSitu.e_confusion = maintaskParam.trial.e_confusion;
						maintaskParam.trialInSitu.e_contempt = maintaskParam.trial.e_contempt;
						maintaskParam.trialInSitu.e_devastation = maintaskParam.trial.e_devastation;
						maintaskParam.trialInSitu.e_disappointment = maintaskParam.trial.e_disappointment;
						maintaskParam.trialInSitu.e_disgust = maintaskParam.trial.e_disgust;
						maintaskParam.trialInSitu.e_embarrassment = maintaskParam.trial.e_embarrassment;
						maintaskParam.trialInSitu.e_envy = maintaskParam.trial.e_envy;
						maintaskParam.trialInSitu.e_excitement = maintaskParam.trial.e_excitement;
						maintaskParam.trialInSitu.e_fury = maintaskParam.trial.e_fury;
						maintaskParam.trialInSitu.e_gratitude = maintaskParam.trial.e_gratitude;
						maintaskParam.trialInSitu.e_guilt = maintaskParam.trial.e_guilt;
						maintaskParam.trialInSitu.e_joy = maintaskParam.trial.e_joy;
						maintaskParam.trialInSitu.e_pride = maintaskParam.trial.e_pride;
						maintaskParam.trialInSitu.e_regret = maintaskParam.trial.e_regret;
						maintaskParam.trialInSitu.e_relief = maintaskParam.trial.e_relief;
						maintaskParam.trialInSitu.e_respect = maintaskParam.trial.e_respect;
						maintaskParam.trialInSitu.e_surprise = maintaskParam.trial.e_surprise;
						maintaskParam.trialInSitu.e_sympathy = maintaskParam.trial.e_sympathy;
					}

					this.data.push(maintaskParam.trial);


					// if (validate_response_filter(maintaskParam.trial) === false && maintaskParam.numComplete > 1) {
					// 	if (debug) { console.log('bbbbbb subjectValid '); }
					// }

					ResetRanges();
				}

				// If subject has completed all trials, update progress bar and
				// show slide to ask for demographic info
				if (maintaskParam.numComplete >= maintaskParam.numTrials) { // If all the trials have been presented
					showSlide("finalQuestion0");
					// Update progress bar
					$('.bar').css('width', Math.round(300.0 * maintaskParam.numComplete / maintaskParam.numTrials) + 'px');
					$("#trial-num").html(maintaskParam.numComplete);
					$("#total-num").html(maintaskParam.numTrials);

					// Otherwise, if trials not completed yet, update progress bar
					// and go to next trial based on the order in which trials are supposed
					// to occur
				} else { // If there are more trials to present
					maintaskParam.trial = maintaskParam.subsetStimOrdered[maintaskParam.numComplete];
					// document.getElementById('value_stem').scrollIntoView({behavior: 'smooth'});
					
					if (!!maintaskParam.storeDataInSitu) {
						maintaskParam.trialInSitu = this.dataInSitu[maintaskParam.currentTrialNum];
					}

					/// document.getElementById("videoStim").src = serverRoot + "dynamics/" + maintaskParam.subsetStimOrdered[maintaskParam.shuffledOrder[maintaskParam.numComplete]].stimulus + "t.mp4";
					/// enablePlayButton();

					// document.getElementById("imageStim").src = serverRoot + stimPath + "statics/" + maintaskParam.trial.stimulus + ".png";
					// document.getElementById("imageStim").src = serverRoot + stimPath + "statics/" + maintaskParam.trial.randStimulusFace + ".png";

					var photoThumbThisIds = ['photoThumbLargeThis', 'photoThumbLargeThis2', 'photoThumbThis_small'];
					photoThumbThisIds.forEach(function(obj_id) {
						var photoThumbThis = document.getElementById(obj_id);
						if (photoThumbThis.childElementCount === 0) {
							photoThumbThis.appendChild(playerImages[maintaskParam.numComplete].cloneNode(true));
						} else {
							photoThumbThis.replaceChild(playerImages[maintaskParam.numComplete].cloneNode(true), photoThumbThis.firstChild);
						}
						if (obj_id === 'photoThumbThis_small') { photoThumbThis.childNodes[0].style = 'width:50px; height:50px;'; }
					});

					// if (maintaskParam.trial.stimulus[4] == 1) {
					// 	document.getElementById("imageStim_front1").src = document.getElementById("photoThumbLargeThis").src;
						document.getElementById("imageStim_front2").src = serverRoot + "images/generic_avatar_male.png";
					// } else if (maintaskParam.trial.stimulus[4] == 2) {
						// document.getElementById("imageStim_front1").src = serverRoot + "images/generic_avatar_male.png";
						// document.getElementById("imageStim_front2").src = document.getElementById("imageStim").src;
					// }


					/// console.log("currentImg", maintaskParam.trial.stimulus);


					$('#context_jackpot').text("$" + numberWithCommas(maintaskParam.trial.pot));
					$('#context_jackpot_front').text("$" + numberWithCommas(maintaskParam.trial.pot));

					$('#contextText_decisionOther').html("&nbsp;" + maintaskParam.trial.decisionOther);
					$('#contextText_decisionThis').html("&nbsp;" + maintaskParam.trial.decisionThis);
					document.getElementById("contextImg_decisionOther").src = serverRoot + "images/" + maintaskParam.trial.decisionOther + "Ball.png";
					document.getElementById("contextImg_decisionThis").src = serverRoot + "images/" + maintaskParam.trial.decisionThis + "Ball.png";
					document.getElementById("miniface_Other").src = serverRoot + "images/generic_avatar_male.png";
					// document.getElementById("miniface_This").src = document.getElementById("imageStim").src;

					var outcomeOther = 0;
					var outcomeThis = 0;
					if (maintaskParam.trial.decisionOther === "Split" && maintaskParam.trial.decisionThis === "Split") {
						outcomeOther = 'Won $' + numberWithCommas(Math.floor(maintaskParam.trial.pot * 50) / 100);
						outcomeThis = outcomeOther;

						preload_named_images.replace("imageContext", "CC"); // document.getElementById("imageContext").src = serverRoot + "images/" + "CC.png";
						$('#context_outcome_front1').html(outcomeOther);
						$('#context_outcome_front2').html(outcomeOther);
					}
					if (maintaskParam.trial.decisionOther === "Split" && maintaskParam.trial.decisionThis === "Stole") {
						outcomeOther = 'Won $0.00';
						outcomeThis = 'Won $' + numberWithCommas(maintaskParam.trial.pot);

						// if (maintaskParam.trial.stimulus[4] == 1) {
						preload_named_images.replace("imageContext", "DC"); // document.getElementById("imageContext").src = serverRoot + "images/" + "DC.png";
							$('#context_outcome_front1').html(outcomeThis);
							$('#context_outcome_front2').html(outcomeOther);
						// } else if (maintaskParam.trial.stimulus[4] == 2) {
							// document.getElementById("imageContext").src = serverRoot + "images/" + "CD.png";
							// $('#context_outcome_front1').html(outcomeOther);
							// $('#context_outcome_front2').html(outcomeThis);
						// }


					}
					if (maintaskParam.trial.decisionOther === "Stole" && maintaskParam.trial.decisionThis === "Split") {
						outcomeOther = 'Won $' + numberWithCommas(maintaskParam.trial.pot);
						outcomeThis = 'Won $0.00';

						// if (maintaskParam.trial.stimulus[4] == 1) {
						preload_named_images.replace("imageContext", "CD"); // document.getElementById("imageContext").src = serverRoot + "images/" + "CD.png";
							$('#context_outcome_front1').html(outcomeThis);
							$('#context_outcome_front2').html(outcomeOther);
						// } else if (maintaskParam.trial.stimulus[4] == 2) {
						// 	document.getElementById("imageContext").src = serverRoot + "images/" + "DC.png";
						// 	$('#context_outcome_front1').html(outcomeOther);
						// 	$('#context_outcome_front2').html(outcomeThis);
						// }

					}
					if (maintaskParam.trial.decisionOther === "Stole" && maintaskParam.trial.decisionThis === "Stole") {
						outcomeOther = 'Won $0.00';
						outcomeThis = 'Won $0.00';

						preload_named_images.replace("imageContext", "DD"); // document.getElementById("imageContext").src = serverRoot + "images/" + "DD.png";
						$('#context_outcome_front1').html(outcomeOther);
						$('#context_outcome_front2').html(outcomeOther);
					}
					$('#context_outcomeOther').html(outcomeOther);
					$('#context_outcomeThis').html(outcomeThis);


					// document.getElementById("player_description").textContent = maintaskParam.trial.desc;
					// document.getElementById("player_description_mini").textContent = maintaskParam.trial.desc;

					// document.querySelectorAll(".contestant_pronoun").forEach(
					// 	function(currentValue, currentIndex, listObj){
					// 		currentValue.textContent = maintaskParam.trial.pronoun;
					// 	}
					// );

					// document.querySelectorAll(".possessive_pronoun").forEach(
					// 	function(currentValue, currentIndex, listObj){
					// 		if (maintaskParam.trial.pronoun == 'he') {
					// 			currentValue.textContent = 'his';
					// 		} else {
					// 			currentValue.textContent = 'her';
					// 		}
					// 	}
					// );

					$("#contextTableFrontDiv").html('&nbsp;');
					$("#contextSubTableID").clone().appendTo("#contextTableFrontDiv"); // insert information in video div

					this.stimulusArray[maintaskParam.numComplete] = maintaskParam.trial.stimulus;
					this.stimDescArray[maintaskParam.numComplete] = maintaskParam.trial.desc;

					console.log(maintaskParam.numComplete + "  --------");
					console.log(maintaskParam.trial.stimulus);
					console.log("This: " + maintaskParam.trial.decisionThis);
					console.log("Other: " + maintaskParam.trial.decisionOther);


					maintaskParam.numComplete++;

					// Update progress bar
					$('.bar').css('width', Math.round(300.0 * maintaskParam.numComplete / maintaskParam.numTrials) + 'px');
					$("#trial-num").html(maintaskParam.numComplete);
					$("#total-num").html(maintaskParam.numTrials);

					showSlide("slideStimulusContext");
				}
				// var startTime = (new Date()).getTime();
				// var endTime = (new Date()).getTime();
				//key = (keyCode == 80) ? "p" : "q",
				//userParity = experiment.keyBindings[key],
				// data = {
				//   stimulus: n,
				//   accuracy: realParity == userParity ? 1 : 0,
				//   rt: endTime - startTime
				// };

				// experiment.data.push(data);
				//setTimeout(experiment.next, 500);

				window.scrollTo(0,0);
				$('#interactionMask').hide();
			}
		};


	}); // end of getJSON
}

// Toggle instructions back and forth
// $("#review-instructions").click(function(){
// 	$('#5').toggle();
// });
