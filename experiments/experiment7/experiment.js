///////////// head /////////////

var debug = false,
	revoke = false;

var shuffledEmotionLabels = shuffleArray(["Amusement", "Annoyance", "Confusion", "Contempt", "Devastation", "Disappointment", "Disgust", "Embarrassment", "Envy", "Excitement", "Fury", "Gratitude", "Guilt", "Joy", "Pride", "Regret", "Relief", "Respect", "Surprise", "Sympathy"]);
randomizeEmotions(shuffledEmotionLabels);

/// initalize sliders ///
$('.not-clicked').mousedown(function() {
	$(this).removeClass('not-clicked').addClass('slider-input');
	$(this).closest('.slider').children('.slider-label-not-clicked').removeClass('slider-label-not-clicked').addClass('slider-label');
});

/// initalize rollover effects of emotion labels ///
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

var iSlide = 0;
var iTrial = 0;
var serverRoot = ""; // requires terminal slash
var stimPath = "../stimuli/"; // requires terminal slash

var subjectValid = true;


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

function ValidateRadios(radioNameList) {
	var pass = false;
	for (var j = 0; j < radioNameList.length; j++) {
		pass = false;
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
	// console.log('entering validation')
	// console.log('responses ' + responses)
	// console.log('expected ' + expected)
	if (responses.length != expected.length) {
		pass = false;
		// console.log('valid size missmatch ' + responses.length + '  ' + expected.length)
	} else {
		for (var i=0; i<expected.length; i++) {
			if (responses[i] != expected[i]) {
				pass = false;
				// console.log(responses[i] + ' vs ' + expected[i]);
			}
		}
	}
	return pass;
}


// Ranges //
function ValidateRanges() {
	var pass = true;
	var unanswered = $("#responsesTable").find(".slider-label-not-clicked");
	if (unanswered.length > 0) {
		pass = false;
		if (!debug) {
			alert("Please provide an answer to all emotions. If you think that a person is not experiencing a given emotion, rate that emotion as --not any-- by moving the sliding marker all the way to the left.");
		} else {
			pass = true;
		}
	}
	return pass;
}

function validateDemoRanges() {
	var pass = true;

	var unanswered = $("#demoTable").find(".slider-label-not-clicked");
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
		if (values.length > max || values.lengt < min) {
			// invalid word number
			return false;
		}
	}
	return valid;
}

var tt = 0;
var t0 = 0;
function timeEvent(t,i) {
	if (i == 1) {
		t = new Date().getTime();
		return t;
	} else {
		var t1 = new Date().getTime() - t;
		return t1/1000;
	}
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


/// Genertic Functions ///

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


/// Window Control ///

function slideUp() {
	iSlide++;
}

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

	t0 = timeEvent(t0,1);
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
	// var tempurl = serverRoot + stimPath + "dynamics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[ stimNum ]].stimulus + "t.mp4";
	fetch(serverRoot + stimPath + "dynamics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[stimNum]].stimulus + "t.mp4").then(function(response) {
		return response.blob();
	}).then(function(data) {
		document.getElementById("videoStim").src = URL.createObjectURL(data);
		enablePlayButton();
		console.log("Current load: stimNum    ", stimNum, "maintaskParam.shuffledOrder[ stimNum ]                  ", maintaskParam.shuffledOrder[stimNum], "stimID", maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[stimNum]].stimulus, "( maintaskParam.numComplete", maintaskParam.numComplete, ")");
		// console.log("maintaskParam.numComplete", maintaskParam.numComplete, "maintaskParam.shuffledOrder[ maintaskParam.numComplete ]", maintaskParam.shuffledOrder[ maintaskParam.numComplete ], "stimID", maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[ maintaskParam.numComplete ]].stimulus);
	}).catch(function() {
		console.log("Booo");
	});
}
*/

function preloadStim(stimNum) {
	var req = new XMLHttpRequest();
	var stimURL = serverRoot + stimPath + "dynamics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[stimNum]].stimulus + "t.mp4";
	req.open('GET', stimURL, true);
	req.responseType = 'blob';

	req.onload = function() {
		// Onload is triggered even on 404
		// so we need to check the status code
		if (this.status === 200) {
			var videoBlob = this.response;
			document.getElementById("videoStim").src = URL.createObjectURL(videoBlob); // IE10+
			enablePlayButton();
			console.log("Current load: stimNum    ", stimNum, "maintaskParam.shuffledOrder[ stimNum ]                  ", maintaskParam.shuffledOrder[stimNum], "stimID", maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[stimNum]].stimulus, "( maintaskParam.numComplete", maintaskParam.numComplete, ")");
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
		var stimURL = serverRoot + stimPath + "dynamics/" + "258_context1_vbr1_H265.mp4";
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

function enableAdvance() {
	$('#training_advance_button').removeClass('advance-button-inactive').addClass('advance-button');
	$('#training_advance_button').removeClass('advance-button-inactive').addClass('advance-button');
	if (!debug) {
		document.getElementById('training_advance_button').onclick = function() {
			if(!!maintask.validate0('bypass')){this.blur(); showSlide(4);}
		};
	} else {
		document.getElementById('training_advance_button').onclick = function() {
			if(!!maintask.validate0('7510')){this.blur(); showSlide(4);}
		};
	}
}

/// Experiment ///

function SetMaintaskParam(selectedTrials) {
	this.allConditions = [
		[
			{ "condition": 1, "Version": "1a", "stimID": 235.1, "stimulus": "235_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 50221.00 }, // 0
			{ "condition": 1, "Version": "1a", "stimID": 235.2, "stimulus": "235_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 50221.00 }, // 1
			{ "condition": 1, "Version": "1a", "stimID": 237.1, "stimulus": "237_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 65673.50 }, // 2
			{ "condition": 1, "Version": "1a", "stimID": 237.2, "stimulus": "237_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 65673.50 }, // 3
			{ "condition": 1, "Version": "1a", "stimID": 239.1, "stimulus": "239_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 94819.00 }, // 4
			{ "condition": 1, "Version": "1a", "stimID": 239.2, "stimulus": "239_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 94819.00 }, // 5
			{ "condition": 1, "Version": "1a", "stimID": 240.1, "stimulus": "240_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 36159.00 }, // 6
			{ "condition": 1, "Version": "1a", "stimID": 240.2, "stimulus": "240_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 36159.00 }, // 7
			{ "condition": 1, "Version": "1a", "stimID": 241.1, "stimulus": "241_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 80954.50 }, // 8
			{ "condition": 1, "Version": "1a", "stimID": 241.2, "stimulus": "241_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 80954.50 }, // 9
			{ "condition": 1, "Version": "1a", "stimID": 242.1, "stimulus": "242_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 1283.50 }, // 10
			{ "condition": 1, "Version": "1a", "stimID": 242.2, "stimulus": "242_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1283.50 }, // 11
			{ "condition": 1, "Version": "1a", "stimID": 243.1, "stimulus": "243_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 7726.50 }, // 12
			{ "condition": 1, "Version": "1a", "stimID": 243.2, "stimulus": "243_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 7726.50 }, // 13
			{ "condition": 1, "Version": "1a", "stimID": 244.1, "stimulus": "244_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 139.00 }, // 14
			{ "condition": 1, "Version": "1a", "stimID": 244.2, "stimulus": "244_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 139.00 }, // 15
			{ "condition": 1, "Version": "1a", "stimID": 245.1, "stimulus": "245_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 5958.00 }, // 16
			{ "condition": 1, "Version": "1a", "stimID": 245.2, "stimulus": "245_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 5958.00 }, // 17
			{ "condition": 1, "Version": "1a", "stimID": 247.1, "stimulus": "247_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 3.50 }, // 18
			{ "condition": 1, "Version": "1a", "stimID": 247.2, "stimulus": "247_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 3.50 }, // 19
			{ "condition": 1, "Version": "1a", "stimID": 248.1, "stimulus": "248_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 31403.00 }, // 20
			{ "condition": 1, "Version": "1a", "stimID": 248.2, "stimulus": "248_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 31403.00 }, // 21
			{ "condition": 1, "Version": "1a", "stimID": 249.1, "stimulus": "249_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 1562.50 }, // 22
			{ "condition": 1, "Version": "1a", "stimID": 249.2, "stimulus": "249_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 1562.50 }, // 23
			{ "condition": 1, "Version": "1a", "stimID": 250.1, "stimulus": "250_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 30304.50 }, // 24
			{ "condition": 1, "Version": "1a", "stimID": 250.2, "stimulus": "250_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 30304.50 }, // 25
			{ "condition": 1, "Version": "1a", "stimID": 251.1, "stimulus": "251_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 2835.50 }, // 26
			{ "condition": 1, "Version": "1a", "stimID": 251.2, "stimulus": "251_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 2835.50 }, // 27
			{ "condition": 1, "Version": "1a", "stimID": 252.1, "stimulus": "252_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 310.00 }, // 28
			{ "condition": 1, "Version": "1a", "stimID": 252.2, "stimulus": "252_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 310.00 }, // 29
			{ "condition": 1, "Version": "1a", "stimID": 253.1, "stimulus": "253_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 21719.50 }, // 30
			{ "condition": 1, "Version": "1a", "stimID": 253.2, "stimulus": "253_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 21719.50 }, // 31
			{ "condition": 1, "Version": "1a", "stimID": 254.1, "stimulus": "254_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 130884.00 }, // 32
			{ "condition": 1, "Version": "1a", "stimID": 254.2, "stimulus": "254_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 130884.00 }, // 33
			{ "condition": 1, "Version": "1a", "stimID": 256.1, "stimulus": "256_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 2983.00 }, // 34
			{ "condition": 1, "Version": "1a", "stimID": 256.2, "stimulus": "256_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 2983.00 }, // 35
			{ "condition": 1, "Version": "1a", "stimID": 257.1, "stimulus": "257_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 30.00 }, // 36
			{ "condition": 1, "Version": "1a", "stimID": 257.2, "stimulus": "257_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 30.00 }, // 37
			{ "condition": 1, "Version": "1a", "stimID": 260.1, "stimulus": "260_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 1598.50 }, // 38
			{ "condition": 1, "Version": "1a", "stimID": 260.2, "stimulus": "260_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 1598.50 }, // 39
			{ "condition": 1, "Version": "1a", "stimID": 262.1, "stimulus": "262_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 56488.00 }, // 40
			{ "condition": 1, "Version": "1a", "stimID": 262.2, "stimulus": "262_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 56488.00 }, // 41
			{ "condition": 1, "Version": "1a", "stimID": 263.1, "stimulus": "263_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 28802.00 }, // 42
			{ "condition": 1, "Version": "1a", "stimID": 263.2, "stimulus": "263_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 28802.00 }, // 43
			{ "condition": 1, "Version": "1a", "stimID": 264.1, "stimulus": "264_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 24145.00 }, // 44
			{ "condition": 1, "Version": "1a", "stimID": 264.2, "stimulus": "264_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 24145.00 }, // 45
			{ "condition": 1, "Version": "1a", "stimID": 265.1, "stimulus": "265_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 1588.00 }, // 46
			{ "condition": 1, "Version": "1a", "stimID": 265.2, "stimulus": "265_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1588.00 }, // 47
			{ "condition": 1, "Version": "1a", "stimID": 266.1, "stimulus": "266_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 6559.00 }, // 48
			{ "condition": 1, "Version": "1a", "stimID": 266.2, "stimulus": "266_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 6559.00 }, // 49
			{ "condition": 1, "Version": "1a", "stimID": 268.1, "stimulus": "268_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 61381.50 }, // 50
			{ "condition": 1, "Version": "1a", "stimID": 268.2, "stimulus": "268_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 61381.50 }, // 51
			{ "condition": 1, "Version": "1a", "stimID": 269.1, "stimulus": "269_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 2318.00 }, // 52
			{ "condition": 1, "Version": "1a", "stimID": 269.2, "stimulus": "269_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 2318.00 }, // 53
			{ "condition": 1, "Version": "1a", "stimID": 270.1, "stimulus": "270_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 5322.50 }, // 54
			{ "condition": 1, "Version": "1a", "stimID": 270.2, "stimulus": "270_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 5322.50 }, // 55
			{ "condition": 1, "Version": "1a", "stimID": 271.1, "stimulus": "271_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 19025.50 }, // 56
			{ "condition": 1, "Version": "1a", "stimID": 271.2, "stimulus": "271_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 19025.50 }, // 57
			{ "condition": 1, "Version": "1a", "stimID": 272.1, "stimulus": "272_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 81899.00 }, // 58
			{ "condition": 1, "Version": "1a", "stimID": 272.2, "stimulus": "272_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 81899.00 }, // 59
			{ "condition": 1, "Version": "1a", "stimID": 273.1, "stimulus": "273_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 9675.00 }, // 60
			{ "condition": 1, "Version": "1a", "stimID": 273.2, "stimulus": "273_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 9675.00 }, // 61
			{ "condition": 1, "Version": "1a", "stimID": 276.1, "stimulus": "276_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 130944.00 }, // 62
			{ "condition": 1, "Version": "1a", "stimID": 276.2, "stimulus": "276_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 130944.00 }, // 63
			{ "condition": 1, "Version": "1a", "stimID": 277.1, "stimulus": "277_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 3331.00 }, // 64
			{ "condition": 1, "Version": "1a", "stimID": 277.2, "stimulus": "277_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 3331.00 }, // 65
			{ "condition": 1, "Version": "1a", "stimID": 278.1, "stimulus": "278_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 40606.00 }, // 66
			{ "condition": 1, "Version": "1a", "stimID": 278.2, "stimulus": "278_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 40606.00 }, // 67
			{ "condition": 1, "Version": "1a", "stimID": 279.1, "stimulus": "279_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 269.50 }, // 68
			{ "condition": 1, "Version": "1a", "stimID": 279.2, "stimulus": "279_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 269.50 }, // 69
			{ "condition": 1, "Version": "1a", "stimID": 281.1, "stimulus": "281_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 1803.00 }, // 70
			{ "condition": 1, "Version": "1a", "stimID": 281.2, "stimulus": "281_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1803.00 }, // 71
			{ "condition": 1, "Version": "1a", "stimID": 282.1, "stimulus": "282_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 2843.50 }, // 72
			{ "condition": 1, "Version": "1a", "stimID": 282.2, "stimulus": "282_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 2843.50 }, // 73
			{ "condition": 1, "Version": "1a", "stimID": 283.1, "stimulus": "283_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 57518.00 }, // 74
			{ "condition": 1, "Version": "1a", "stimID": 283.2, "stimulus": "283_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 57518.00 }, // 75
			{ "condition": 1, "Version": "1a", "stimID": 284.1, "stimulus": "284_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1116.50 }, // 76
			{ "condition": 1, "Version": "1a", "stimID": 284.2, "stimulus": "284_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 1116.50 }, // 77
			{ "condition": 1, "Version": "1a", "stimID": 285.1, "stimulus": "285_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 2300.50 }, // 78
			{ "condition": 1, "Version": "1a", "stimID": 285.2, "stimulus": "285_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 2300.50 }, // 79
			{ "condition": 1, "Version": "1a", "stimID": 286.1, "stimulus": "286_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 3532.50 }, // 80
			{ "condition": 1, "Version": "1a", "stimID": 286.2, "stimulus": "286_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 3532.50 }, // 81
			{ "condition": 1, "Version": "1a", "stimID": 287.1, "stimulus": "287_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 1030.50 }, // 82
			{ "condition": 1, "Version": "1a", "stimID": 287.2, "stimulus": "287_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1030.50 }, // 83
			{ "condition": 1, "Version": "1a", "stimID": 288.1, "stimulus": "288_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 15744.50 }, // 84
			{ "condition": 1, "Version": "1a", "stimID": 288.2, "stimulus": "288_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 15744.50 }, // 85
			{ "condition": 1, "Version": "1a", "stimID": 289.1, "stimulus": "289_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 822.50 }, // 86
			{ "condition": 1, "Version": "1a", "stimID": 289.2, "stimulus": "289_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 822.50 } // 87
		],
		[
			{ "condition": 2, "Version": "1a", "stimID": "001.1", "stimulus": "001_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 2.00 }, // 0
			{ "condition": 2, "Version": "1a", "stimID": "001.2", "stimulus": "001_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 2.00 }, // 1
			{ "condition": 2, "Version": "1a", "stimID": "001.3", "stimulus": "001_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 2.00 }, // 2
			{ "condition": 2, "Version": "1a", "stimID": "001.4", "stimulus": "001_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 2.00 }, // 3
			{ "condition": 2, "Version": "1a", "stimID": "002.1", "stimulus": "002_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 11.00 }, // 4
			{ "condition": 2, "Version": "1a", "stimID": "002.2", "stimulus": "002_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 11.00 }, // 5
			{ "condition": 2, "Version": "1a", "stimID": "002.3", "stimulus": "002_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 11.00 }, // 6
			{ "condition": 2, "Version": "1a", "stimID": "002.4", "stimulus": "002_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 11.00 }, // 7
			{ "condition": 2, "Version": "1a", "stimID": "003.1", "stimulus": "003_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 25.00 }, // 8
			{ "condition": 2, "Version": "1a", "stimID": "003.2", "stimulus": "003_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 25.00 }, // 9
			{ "condition": 2, "Version": "1a", "stimID": "003.3", "stimulus": "003_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 25.00 }, // 10
			{ "condition": 2, "Version": "1a", "stimID": "003.4", "stimulus": "003_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 25.00 }, // 11
			{ "condition": 2, "Version": "1a", "stimID": "004.1", "stimulus": "004_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 46.00 }, // 12
			{ "condition": 2, "Version": "1a", "stimID": "004.2", "stimulus": "004_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 46.00 }, // 13
			{ "condition": 2, "Version": "1a", "stimID": "004.3", "stimulus": "004_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 46.00 }, // 14
			{ "condition": 2, "Version": "1a", "stimID": "004.4", "stimulus": "004_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 46.00 }, // 15
			{ "condition": 2, "Version": "1a", "stimID": "005.1", "stimulus": "005_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 77.00 }, // 16
			{ "condition": 2, "Version": "1a", "stimID": "005.2", "stimulus": "005_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 77.00 }, // 17
			{ "condition": 2, "Version": "1a", "stimID": "005.3", "stimulus": "005_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 77.00 }, // 18
			{ "condition": 2, "Version": "1a", "stimID": "005.4", "stimulus": "005_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 77.00 }, // 19
			{ "condition": 2, "Version": "1a", "stimID": "006.1", "stimulus": "006_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 124.00 }, // 20
			{ "condition": 2, "Version": "1a", "stimID": "006.2", "stimulus": "006_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 124.00 }, // 21
			{ "condition": 2, "Version": "1a", "stimID": "006.3", "stimulus": "006_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 124.00 }, // 22
			{ "condition": 2, "Version": "1a", "stimID": "006.4", "stimulus": "006_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 124.00 }, // 23
			{ "condition": 2, "Version": "1a", "stimID": "007.1", "stimulus": "007_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 194.00 }, // 24
			{ "condition": 2, "Version": "1a", "stimID": "007.2", "stimulus": "007_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 194.00 }, // 25
			{ "condition": 2, "Version": "1a", "stimID": "007.3", "stimulus": "007_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 194.00 }, // 26
			{ "condition": 2, "Version": "1a", "stimID": "007.4", "stimulus": "007_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 194.00 }, // 27
			{ "condition": 2, "Version": "1a", "stimID": "008.1", "stimulus": "008_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 299.00 }, // 28
			{ "condition": 2, "Version": "1a", "stimID": "008.2", "stimulus": "008_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 299.00 }, // 29
			{ "condition": 2, "Version": "1a", "stimID": "008.3", "stimulus": "008_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 299.00 }, // 30
			{ "condition": 2, "Version": "1a", "stimID": "008.4", "stimulus": "008_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 299.00 }, // 31
			{ "condition": 2, "Version": "1a", "stimID": "009.1", "stimulus": "009_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 457.00 }, // 32
			{ "condition": 2, "Version": "1a", "stimID": "009.2", "stimulus": "009_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 457.00 }, // 33
			{ "condition": 2, "Version": "1a", "stimID": "009.3", "stimulus": "009_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 457.00 }, // 34
			{ "condition": 2, "Version": "1a", "stimID": "009.4", "stimulus": "009_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 457.00 }, // 35
			{ "condition": 2, "Version": "1a", "stimID": "010.1", "stimulus": "010_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 694.00 }, // 36
			{ "condition": 2, "Version": "1a", "stimID": "010.2", "stimulus": "010_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 694.00 }, // 37
			{ "condition": 2, "Version": "1a", "stimID": "010.3", "stimulus": "010_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 694.00 }, // 38
			{ "condition": 2, "Version": "1a", "stimID": "010.4", "stimulus": "010_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 694.00 }, // 39
			{ "condition": 2, "Version": "1a", "stimID": "011.1", "stimulus": "011_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 1049.00 }, // 40
			{ "condition": 2, "Version": "1a", "stimID": "011.2", "stimulus": "011_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 1049.00 }, // 41
			{ "condition": 2, "Version": "1a", "stimID": "011.3", "stimulus": "011_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1049.00 }, // 42
			{ "condition": 2, "Version": "1a", "stimID": "011.4", "stimulus": "011_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 1049.00 }, // 43
			{ "condition": 2, "Version": "1a", "stimID": "012.1", "stimulus": "012_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 1582.00 }, // 44
			{ "condition": 2, "Version": "1a", "stimID": "012.2", "stimulus": "012_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 1582.00 }, // 45
			{ "condition": 2, "Version": "1a", "stimID": "012.3", "stimulus": "012_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1582.00 }, // 46
			{ "condition": 2, "Version": "1a", "stimID": "012.4", "stimulus": "012_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 1582.00 }, // 47
			{ "condition": 2, "Version": "1a", "stimID": "013.1", "stimulus": "013_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 2381.00 }, // 48
			{ "condition": 2, "Version": "1a", "stimID": "013.2", "stimulus": "013_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 2381.00 }, // 49
			{ "condition": 2, "Version": "1a", "stimID": "013.3", "stimulus": "013_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 2381.00 }, // 50
			{ "condition": 2, "Version": "1a", "stimID": "013.4", "stimulus": "013_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 2381.00 }, // 51
			{ "condition": 2, "Version": "1a", "stimID": "014.1", "stimulus": "014_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 3580.00 }, // 52
			{ "condition": 2, "Version": "1a", "stimID": "014.2", "stimulus": "014_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 3580.00 }, // 53
			{ "condition": 2, "Version": "1a", "stimID": "014.3", "stimulus": "014_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 3580.00 }, // 54
			{ "condition": 2, "Version": "1a", "stimID": "014.4", "stimulus": "014_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 3580.00 }, // 55
			{ "condition": 2, "Version": "1a", "stimID": "015.1", "stimulus": "015_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 5378.00 }, // 56
			{ "condition": 2, "Version": "1a", "stimID": "015.2", "stimulus": "015_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 5378.00 }, // 57
			{ "condition": 2, "Version": "1a", "stimID": "015.3", "stimulus": "015_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 5378.00 }, // 58
			{ "condition": 2, "Version": "1a", "stimID": "015.4", "stimulus": "015_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 5378.00 }, // 59
			{ "condition": 2, "Version": "1a", "stimID": "016.1", "stimulus": "016_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 8075.00 }, // 60
			{ "condition": 2, "Version": "1a", "stimID": "016.2", "stimulus": "016_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 8075.00 }, // 61
			{ "condition": 2, "Version": "1a", "stimID": "016.3", "stimulus": "016_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 8075.00 }, // 62
			{ "condition": 2, "Version": "1a", "stimID": "016.4", "stimulus": "016_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 8075.00 }, // 63
			{ "condition": 2, "Version": "1a", "stimID": "017.1", "stimulus": "017_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 12121.00 }, // 64
			{ "condition": 2, "Version": "1a", "stimID": "017.2", "stimulus": "017_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 12121.00 }, // 65
			{ "condition": 2, "Version": "1a", "stimID": "017.3", "stimulus": "017_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 12121.00 }, // 66
			{ "condition": 2, "Version": "1a", "stimID": "017.4", "stimulus": "017_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 12121.00 }, // 67
			{ "condition": 2, "Version": "1a", "stimID": "018.1", "stimulus": "018_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 18190.00 }, // 68
			{ "condition": 2, "Version": "1a", "stimID": "018.2", "stimulus": "018_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 18190.00 }, // 69
			{ "condition": 2, "Version": "1a", "stimID": "018.3", "stimulus": "018_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 18190.00 }, // 70
			{ "condition": 2, "Version": "1a", "stimID": "018.4", "stimulus": "018_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 18190.00 }, // 71
			{ "condition": 2, "Version": "1a", "stimID": "019.1", "stimulus": "019_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 27293.00 }, // 72
			{ "condition": 2, "Version": "1a", "stimID": "019.2", "stimulus": "019_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 27293.00 }, // 73
			{ "condition": 2, "Version": "1a", "stimID": "019.3", "stimulus": "019_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 27293.00 }, // 74
			{ "condition": 2, "Version": "1a", "stimID": "019.4", "stimulus": "019_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 27293.00 }, // 75
			{ "condition": 2, "Version": "1a", "stimID": "020.1", "stimulus": "020_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 40948.00 }, // 76
			{ "condition": 2, "Version": "1a", "stimID": "020.2", "stimulus": "020_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 40948.00 }, // 77
			{ "condition": 2, "Version": "1a", "stimID": "020.3", "stimulus": "020_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 40948.00 }, // 78
			{ "condition": 2, "Version": "1a", "stimID": "020.4", "stimulus": "020_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 40948.00 }, // 79
			{ "condition": 2, "Version": "1a", "stimID": "021.1", "stimulus": "021_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 61430.00 }, // 80
			{ "condition": 2, "Version": "1a", "stimID": "021.2", "stimulus": "021_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 61430.00 }, // 81
			{ "condition": 2, "Version": "1a", "stimID": "021.3", "stimulus": "021_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 61430.00 }, // 82
			{ "condition": 2, "Version": "1a", "stimID": "021.4", "stimulus": "021_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 61430.00 }, // 83
			{ "condition": 2, "Version": "1a", "stimID": "022.1", "stimulus": "022_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 92153.00 }, // 84
			{ "condition": 2, "Version": "1a", "stimID": "022.2", "stimulus": "022_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 92153.00 }, // 85
			{ "condition": 2, "Version": "1a", "stimID": "022.3", "stimulus": "022_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 92153.00 }, // 86
			{ "condition": 2, "Version": "1a", "stimID": "022.4", "stimulus": "022_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 92153.00 }, // 87
			{ "condition": 2, "Version": "1a", "stimID": "023.1", "stimulus": "023_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 138238.00 }, // 88
			{ "condition": 2, "Version": "1a", "stimID": "023.2", "stimulus": "023_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 138238.00 }, // 89
			{ "condition": 2, "Version": "1a", "stimID": "023.3", "stimulus": "023_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 138238.00 }, // 90
			{ "condition": 2, "Version": "1a", "stimID": "023.4", "stimulus": "023_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 138238.00 }, // 91
			{ "condition": 2, "Version": "1a", "stimID": "024.1", "stimulus": "024_cc", "decisionThis": "Split", "decisionOther": "Split", "pot": 207365.00 }, // 92
			{ "condition": 2, "Version": "1a", "stimID": "024.2", "stimulus": "024_cd", "decisionThis": "Split", "decisionOther": "Stole", "pot": 207365.00 }, // 93
			{ "condition": 2, "Version": "1a", "stimID": "024.3", "stimulus": "024_dc", "decisionThis": "Stole", "decisionOther": "Split", "pot": 207365.00 }, // 94
			{ "condition": 2, "Version": "1a", "stimID": "024.4", "stimulus": "024_dd", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 207365.00 } // 95
		]
	];

	/* Experimental Variables */
	// Number of conditions in experiment
	this.numConditions = 1; //allConditions.length;

	// Randomly select a condition number for this particular participant
	this.chooseCondition = 2; // random(0, numConditions-1);

	// Based on condition number, choose set of input (trials)
	this.allTrialOrders = this.allConditions[this.chooseCondition - 1];

	// Produce random order in which the trials will occur
	// this.shuffledOrder = shuffleArray(genIntRange(0, this.allTrialOrders.length - 1));
	this.shuffledOrder = shuffleArray(selectedTrials);
	console.log('shuffledOrder', this.shuffledOrder);

	// Number of trials in each condition
	this.numTrials = this.shuffledOrder.length; //not necessarily this.allTrialOrders.length;

	// Pull the random subet
	this.subsetTrialOrders = [];
	for (var i = 0; i < this.numTrials; i++) {
		this.subsetTrialOrders.push(this.allTrialOrders[i]);
	}

	// Keep track of current trial 
	this.currentTrialNum = 0;

	// Keep track of how many trials have been completed
	this.numComplete = 0;

	this.storeDataInSitu = false;

	var randStimulusFace = shuffleArray(["244_1", "250_1", "271_1", "272_1", "275_1", "283_1", "286_1", "288_1"]);
	// console.log(randStimulusFace.length);
	for (var j = 0; j < randStimulusFace.length; j++) {
		// this.allTrialOrders[j].push({key:"randStimulusFace", value:randStimulusFace[j]});
		$.extend(this.allTrialOrders[this.shuffledOrder[j]], {"randStimulusFace": randStimulusFace[j]} );
	}
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

var phpParam = { baseURL: 'https://daeda.scripts.mit.edu/serveCondition/serveCondition.php?callback=?', nReps: 1, condfname: "servedConditions.csv" };
var maintask = [];
var maintaskParam = [];

function loadHIT(nextSlide) {
	// start experiment timer
	tt = timeEvent(tt,1);

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
	}

	var requestService = 'nReps=' + phpParam.nReps + '&writestatus=' + phpParam.writestatus + '&condComplete=' + 'REQUEST' + '&condfname=' + phpParam.condfname;

	showSlide("loadingHIT");
	$.getJSON(phpParam.baseURL, requestService, function(res) {

		console.log("Served Condition:", res.condNum);

		showSlide(nextSlide);
		var defaultCondNum = 0; // if PHP runs out of options
		var conditions = [
			[2, 12, 17, 33, 47, 52, 70, 87], // 0
			[18, 19, 25, 34, 51, 76, 77, 92], // 1
			[16, 43, 44, 59, 61, 74, 85, 94], // 2
			[1, 22, 38, 48, 65, 67, 80, 91], // 3
			[13, 20, 39, 50, 54, 72, 73, 75], // 4
			[3, 21, 28, 30, 35, 40, 57, 86], // 5
			[8, 31, 32, 41, 46, 66, 83, 89], // 6
			[5, 7, 10, 29, 64, 79, 82, 84], // 7
			[4, 6, 14, 23, 24, 49, 71, 93], // 8
			[0, 9, 15, 27, 53, 62, 68, 78], // 9
			[42, 55, 56, 69, 81, 88, 90, 95], // 10
			[11, 26, 36, 37, 45, 58, 60, 63], // 11

			[6, 13, 39, 58, 64, 65, 68, 95], // 12
			[0, 1, 4, 5, 63, 67, 70, 82], // 13
			[7, 14, 41, 52, 84, 85, 90, 91], // 14
			[10, 16, 19, 23, 34, 60, 89, 93], // 15
			[2, 26, 35, 37, 44, 51, 61, 80], // 16
			[9, 11, 12, 17, 30, 43, 48, 86], // 17
			[8, 31, 36, 50, 69, 74, 77, 87], // 18
			[24, 25, 38, 54, 55, 73, 76, 79], // 19
			[18, 22, 47, 53, 72, 75, 81, 92], // 20
			[3, 15, 20, 29, 46, 56, 57, 78], // 21
			[27, 32, 45, 49, 62, 83, 88, 94], // 22
			[21, 28, 33, 40, 42, 59, 66, 71], // 23

			[4, 11, 21, 30, 33, 51, 70, 92], // 24
			[13, 26, 36, 43, 61, 75, 78, 80], // 25
			[9, 14, 19, 44, 52, 67, 77, 90], // 26
			[18, 20, 32, 35, 50, 57, 59, 81], // 27
			[8, 25, 28, 49, 55, 66, 71, 94], // 28
			[6, 45, 46, 47, 56, 72, 87, 93], // 29
			[2, 12, 23, 31, 54, 60, 65, 85], // 30
			[15, 17, 37, 40, 64, 74, 83, 86], // 31
			[1, 3, 16, 24, 42, 53, 58, 91], // 32
			[27, 38, 48, 63, 68, 73, 82, 89], // 33
			[0, 5, 7, 10, 22, 41, 79, 88], // 34
			[29, 34, 39, 62, 69, 76, 84, 95], // 35

			[16, 25, 27, 58, 60, 73, 91, 94], // 36
			[29, 49, 56, 62, 63, 67, 76, 82], // 37
			[11, 32, 33, 34, 38, 44, 45, 75], // 38
			[5, 8, 24, 39, 50, 55, 61, 70], // 39
			[0, 3, 10, 13, 23, 68, 85, 86], // 40
			[2, 4, 28, 30, 41, 65, 83, 95], // 41
			[7, 18, 20, 21, 71, 72, 81, 90], // 42
			[6, 14, 37, 40, 59, 64, 87, 93], // 43
			[15, 26, 36, 51, 53, 57, 74, 92], // 44
			[17, 19, 22, 52, 69, 78, 79, 84], // 45
			[9, 12, 35, 42, 47, 54, 77, 88], // 46
			[1, 31, 43, 46, 48, 66, 80, 89], // 47

			[9, 21, 27, 32, 43, 46, 82, 92], // 48
			[2, 17, 20, 31, 33, 64, 66, 71], // 49
			[11, 18, 30, 40, 41, 63, 81, 84], // 50
			[14, 16, 59, 61, 76, 79, 89, 94], // 51
			[12, 37, 38, 50, 51, 57, 72, 91], // 52
			[7, 26, 36, 56, 58, 65, 75, 93], // 53
			[1, 29, 35, 48, 62, 78, 80, 87], // 54
			[0, 4, 5, 15, 22, 47, 73, 74], // 55
			[13, 19, 28, 42, 45, 52, 83, 86], // 56
			[6, 8, 24, 34, 39, 49, 53, 55], // 57
			[10, 54, 67, 68, 77, 85, 88, 95], // 58
			[3, 23, 25, 44, 60, 69, 70, 90], // 59

			[16, 27, 47, 58, 61, 78, 88, 93], // 60
			[2, 24, 25, 32, 50, 55, 69, 91], // 61
			[1, 4, 9, 31, 48, 54, 62, 71], // 62
			[13, 17, 30, 35, 40, 75, 86, 92], // 63
			[3, 6, 23, 28, 49, 53, 68, 70], // 64
			[12, 26, 33, 43, 64, 65, 79, 90], // 65
			[0, 7, 18, 29, 38, 39, 73, 80], // 66
			[8, 19, 44, 74, 82, 83, 85, 89], // 67
			[10, 14, 52, 57, 59, 67, 72, 81], // 68
			[11, 15, 21, 22, 60, 66, 76, 77], // 69
			[37, 41, 46, 56, 84, 87, 94, 95], // 70
			[5, 20, 34, 36, 42, 45, 51, 63], // 71

			[30, 35, 44, 47, 49, 53, 78, 80], // 72
			[7, 12, 15, 18, 21, 61, 62, 84], // 73
			[1, 2, 3, 17, 22, 40, 48, 95], // 74
			[14, 19, 52, 59, 60, 81, 85, 86], // 75
			[4, 10, 27, 29, 41, 68, 71, 90], // 76
			[20, 38, 45, 55, 72, 73, 79, 94], // 77
			[9, 25, 26, 32, 36, 46, 83, 91], // 78
			[8, 11, 33, 37, 39, 58, 70, 76], // 79
			[13, 28, 42, 63, 69, 82, 87, 88], // 80
			[0, 34, 43, 51, 64, 65, 66, 77], // 81
			[6, 23, 24, 31, 74, 89, 92, 93], // 82
			[5, 16, 50, 54, 56, 57, 67, 75], // 83

			[18, 23, 27, 33, 53, 64, 68, 82], // 84
			[6, 15, 37, 38, 43, 88, 89, 92], // 85
			[9, 16, 32, 34, 35, 55, 58, 81], // 86
			[41, 42, 56, 59, 70, 80, 85, 91], // 87
			[7, 13, 40, 51, 84, 90, 93, 94], // 88
			[8, 19, 25, 31, 44, 61, 74, 86], // 89
			[1, 11, 20, 36, 46, 66, 69, 87], // 90
			[2, 29, 45, 47, 52, 62, 67, 72], // 91
			[0, 12, 14, 17, 39, 50, 57, 71], // 92
			[4, 5, 10, 21, 22, 48, 63, 75], // 93
			[24, 26, 30, 49, 76, 77, 79, 95], // 94
			[3, 28, 54, 60, 65, 73, 78, 83], // 95

			[10, 15, 36, 44, 53, 54, 65, 79], // 96
			[0, 20, 29, 34, 41, 59, 83, 94], // 97
			[24, 26, 45, 50, 60, 63, 81, 91], // 98
			[28, 30, 33, 37, 56, 67, 71, 90], // 99
			[2, 16, 27, 38, 47, 73, 80, 93], // 100
			[14, 19, 25, 48, 70, 76, 89, 95], // 101
			[1, 9, 12, 18, 23, 51, 68, 78], // 102
			[7, 11, 17, 32, 66, 85, 86, 88], // 103
			[40, 43, 49, 55, 69, 74, 82, 92], // 104
			[3, 4, 5, 13, 42, 46, 75, 84], // 105
			[31, 52, 58, 61, 62, 64, 77, 87], // 106
			[6, 8, 21, 22, 35, 39, 57, 72], // 107

			[32, 42, 55, 64, 67, 69, 73, 86], // 108
			[4, 8, 11, 18, 29, 57, 58, 63], // 109
			[0, 6, 7, 25, 40, 49, 66, 87], // 110
			[19, 34, 41, 46, 60, 61, 68, 95], // 111
			[3, 14, 38, 39, 45, 52, 81, 88], // 112
			[1, 9, 27, 36, 47, 48, 74, 78], // 113
			[2, 13, 23, 30, 72, 80, 83, 89], // 114
			[28, 56, 65, 70, 79, 85, 91, 94], // 115
			[15, 16, 17, 22, 31, 53, 62, 84], // 116
			[43, 54, 71, 76, 77, 82, 92, 93], // 117
			[20, 21, 24, 26, 37, 59, 75, 90], // 118
			[5, 10, 12, 33, 35, 44, 50, 51], // 119

			[8, 18, 26, 33, 52, 63, 73, 95], // 120
			[5, 24, 36, 37, 55, 78, 87, 94], // 121
			[27, 56, 57, 70, 84, 86, 91, 93], // 122
			[2, 11, 16, 30, 32, 45, 59, 65], // 123
			[15, 34, 40, 41, 64, 67, 82, 89], // 124
			[10, 17, 22, 23, 28, 47, 48, 77], // 125
			[9, 13, 19, 35, 38, 44, 46, 80], // 126
			[31, 49, 50, 54, 61, 75, 76, 88], // 127
			[0, 3, 7, 12, 14, 21, 29, 58], // 128
			[1, 51, 66, 68, 71, 74, 81, 92], // 129
			[20, 39, 42, 60, 69, 83, 85, 90], // 130
			[4, 6, 25, 43, 53, 62, 72, 79] // 131
		];

		var condNum = parseInt(res.condNum);
		var selectedTrials = [];
		if (condNum >= 0 && condNum <= conditions.length - 1) {
			selectedTrials = conditions[condNum];
		} else {
			selectedTrials = conditions[defaultCondNum];
			condNum = defaultCondNum * -1;
		}

		maintaskParam = new SetMaintaskParam(selectedTrials);

		// Updates the progress bar
		$("#trial-num").html(maintaskParam.numComplete);
		$("#total-num").html(maintaskParam.numTrials);


		// var maintaskParam = new SetMaintaskParam();

		maintask = {

			respTimer: new Array(maintaskParam.numTrials),
			stimIDArray: new Array(maintaskParam.numTrials),
			stimulusArray: new Array(maintaskParam.numTrials),
			randFaceId: new Array(maintaskParam.numTrials),

			q1responseArray: new Array(maintaskParam.numTrials),
			q2responseArray: new Array(maintaskParam.numTrials),
			q3responseArray: new Array(maintaskParam.numTrials),
			q4responseArray: new Array(maintaskParam.numTrials),
			q5responseArray: new Array(maintaskParam.numTrials),
			q6responseArray: new Array(maintaskParam.numTrials),
			q7responseArray: new Array(maintaskParam.numTrials),
			q8responseArray: new Array(maintaskParam.numTrials),
			q9responseArray: new Array(maintaskParam.numTrials),
			q10responseArray: new Array(maintaskParam.numTrials),
			q11responseArray: new Array(maintaskParam.numTrials),
			q12responseArray: new Array(maintaskParam.numTrials),
			q13responseArray: new Array(maintaskParam.numTrials),
			q14responseArray: new Array(maintaskParam.numTrials),
			q15responseArray: new Array(maintaskParam.numTrials),
			q16responseArray: new Array(maintaskParam.numTrials),
			q17responseArray: new Array(maintaskParam.numTrials),
			q18responseArray: new Array(maintaskParam.numTrials),
			q19responseArray: new Array(maintaskParam.numTrials),
			q20responseArray: new Array(maintaskParam.numTrials),

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

			expTime: new Array(1),
			windowDims: new Array(0),
			screenDims: new Array(0),

			data: [],
			dataInSitu: [],

			validate0: function(expectedResponse) {
				
				var radios = document.getElementsByName('v0');
				var radiosValue = false;

				for (var i = 0; i < radios.length; i++) {
					if (radios[i].checked == true) {
						radiosValue = true;
					}
				}
				if (!radiosValue) {
					alert("Please watch the video and answer the question");
					return false;
				} else {
					this.validationRadioExpectedResp[0] = expectedResponse;
					this.validationRadio[0] = $('input[name="v0"]:checked').val();
					return true;
				}
			},

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
				this.expTime = timeEvent(tt);

				subjectValid = validateResponses(this.validationRadio, ['7510','disdain','jealousy','AF25HAS','steal']);
				// if (!!subjectValid) {
				// SEND DATA TO TURK
				// }

				this.dem_gender = $('input[name="d1"]:checked').val();
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
				var returnServe = 'nReps=' + phpParam.nReps + '&writestatus=' + phpParam.writestatus + '&condComplete=' + maintask.randCondNum.toString() + '&subjValid=' + subjectValid.toString().toUpperCase() + '&condfname=' + phpParam.condfname;
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
				showSlide("slideStimulusContext");
				// try {
				// var url = URL.revokeObjectURL(document.getElementById("videoStim_small").src); // IE10+
				// } catch (err) {}
				$('#interactionMask').hide();

				// duplicate allTrialOrders
				if (maintaskParam.numComplete === 0) {
					// this.randCondNum.push(condNum); // push randomization number
					this.randCondNum = condNum;
					// disablePlayButton();
					// preloadStim(0); // load first video

					this.windowDims = [window.innerWidth, window.innerHeight];
					this.screenDims = [window.screen.width, window.screen.height];

					if (!!maintaskParam.storeDataInSitu) {
						for (var i = 0; i < maintaskParam.allTrialOrders.length; i++) {
							var temp = maintaskParam.allTrialOrders[i];
							temp.q1responseArray = "";
							temp.q2responseArray = "";
							temp.q3responseArray = "";
							temp.q4responseArray = "";
							temp.q5responseArray = "";
							temp.q6responseArray = "";
							temp.q7responseArray = "";
							temp.q8responseArray = "";
							temp.q9responseArray = "";
							temp.q10responseArray = "";
							temp.q11responseArray = "";
							temp.q12responseArray = "";
							temp.q13responseArray = "";
							temp.q14responseArray = "";
							temp.q15responseArray = "";
							temp.q16responseArray = "";
							temp.q17responseArray = "";
							temp.q18responseArray = "";
							temp.q19responseArray = "";
							temp.q20responseArray = "";

							this.dataInSitu.push(temp);
							//// CLEAN this up a little?
						}
					}
				}

				// If this is not the first trial, record variables
				if (maintaskParam.numComplete > 0) {

					// get timer count
					maintaskParam.trial.respTimer = timeEvent(t0);
					this.respTimer[maintaskParam.numComplete - 1] = maintaskParam.trial.respTimer;
					// console.log('timer(' + maintaskParam.numComplete + '): ' + this.respTimer[maintaskParam.numComplete - 1]);

					maintaskParam.trial.q1responseArray = e_amusement.value;
					maintaskParam.trial.q2responseArray = e_annoyance.value;
					maintaskParam.trial.q3responseArray = e_confusion.value;
					maintaskParam.trial.q4responseArray = e_contempt.value;
					maintaskParam.trial.q5responseArray = e_devastation.value;
					maintaskParam.trial.q6responseArray = e_disappointment.value;
					maintaskParam.trial.q7responseArray = e_disgust.value;
					maintaskParam.trial.q8responseArray = e_embarrassment.value;
					maintaskParam.trial.q9responseArray = e_envy.value;
					maintaskParam.trial.q10responseArray = e_excitement.value;
					maintaskParam.trial.q11responseArray = e_fury.value;
					maintaskParam.trial.q12responseArray = e_gratitude.value;
					maintaskParam.trial.q13responseArray = e_guilt.value;
					maintaskParam.trial.q14responseArray = e_joy.value;
					maintaskParam.trial.q15responseArray = e_pride.value;
					maintaskParam.trial.q16responseArray = e_regret.value;
					maintaskParam.trial.q17responseArray = e_relief.value;
					maintaskParam.trial.q18responseArray = e_respect.value;
					maintaskParam.trial.q19responseArray = e_surprise.value;
					maintaskParam.trial.q20responseArray = e_sympathy.value;

					this.q1responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q1responseArray;
					this.q2responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q2responseArray;
					this.q3responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q3responseArray;
					this.q4responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q4responseArray;
					this.q5responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q5responseArray;
					this.q6responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q6responseArray;
					this.q7responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q7responseArray;
					this.q8responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q8responseArray;
					this.q9responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q9responseArray;
					this.q10responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q10responseArray;
					this.q11responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q11responseArray;
					this.q12responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q12responseArray;
					this.q13responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q13responseArray;
					this.q14responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q14responseArray;
					this.q15responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q15responseArray;
					this.q16responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q16responseArray;
					this.q17responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q17responseArray;
					this.q18responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q18responseArray;
					this.q19responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q19responseArray;
					this.q20responseArray[maintaskParam.numComplete - 1] = maintaskParam.trial.q20responseArray;

					if (!!maintaskParam.storeDataInSitu) {
						maintaskParam.trialInSitu.q1responseArray = maintaskParam.trial.q1responseArray;
						maintaskParam.trialInSitu.q2responseArray = maintaskParam.trial.q2responseArray;
						maintaskParam.trialInSitu.q3responseArray = maintaskParam.trial.q3responseArray;
						maintaskParam.trialInSitu.q4responseArray = maintaskParam.trial.q4responseArray;
						maintaskParam.trialInSitu.q5responseArray = maintaskParam.trial.q5responseArray;
						maintaskParam.trialInSitu.q6responseArray = maintaskParam.trial.q6responseArray;
						maintaskParam.trialInSitu.q7responseArray = maintaskParam.trial.q7responseArray;
						maintaskParam.trialInSitu.q8responseArray = maintaskParam.trial.q8responseArray;
						maintaskParam.trialInSitu.q9responseArray = maintaskParam.trial.q9responseArray;
						maintaskParam.trialInSitu.q10responseArray = maintaskParam.trial.q10responseArray;
						maintaskParam.trialInSitu.q11responseArray = maintaskParam.trial.q11responseArray;
						maintaskParam.trialInSitu.q12responseArray = maintaskParam.trial.q12responseArray;
						maintaskParam.trialInSitu.q13responseArray = maintaskParam.trial.q13responseArray;
						maintaskParam.trialInSitu.q14responseArray = maintaskParam.trial.q14responseArray;
						maintaskParam.trialInSitu.q15responseArray = maintaskParam.trial.q15responseArray;
						maintaskParam.trialInSitu.q16responseArray = maintaskParam.trial.q16responseArray;
						maintaskParam.trialInSitu.q17responseArray = maintaskParam.trial.q17responseArray;
						maintaskParam.trialInSitu.q18responseArray = maintaskParam.trial.q18responseArray;
						maintaskParam.trialInSitu.q19responseArray = maintaskParam.trial.q19responseArray;
						maintaskParam.trialInSitu.q20responseArray = maintaskParam.trial.q20responseArray;
					}

					this.data.push(maintaskParam.trial);

					ResetRanges();
				}

				// If subject has completed all trials, update progress bar and
				// show slide to ask for demographic info
				if (maintaskParam.numComplete >= maintaskParam.numTrials) {
					showSlide("finalQuestion0");
					// Update progress bar
					$('.bar').css('width', Math.round(300.0 * maintaskParam.numComplete / maintaskParam.numTrials) + 'px');
					$("#trial-num").html(maintaskParam.numComplete);
					$("#total-num").html(maintaskParam.numTrials);

					// Otherwise, if trials not completed yet, update progress bar
					// and go to next trial based on the order in which trials are supposed
					// to occur
				} else {
					//currentTrialNum is used for randomizing later
					maintaskParam.currentTrialNum = maintaskParam.shuffledOrder[maintaskParam.numComplete]; //numComplete //allTrialOrders[numComplete];
					maintaskParam.trial = maintaskParam.allTrialOrders[maintaskParam.currentTrialNum];
					if (!!maintaskParam.storeDataInSitu) {
						maintaskParam.trialInSitu = this.dataInSitu[maintaskParam.currentTrialNum];
					}

					/// document.getElementById("videoStim").src = serverRoot + "dynamics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[maintaskParam.numComplete]].stimulus + "t.mp4";
					/// enablePlayButton();

					// document.getElementById("imageStim").src = serverRoot + stimPath + "statics/" + maintaskParam.trial.stimulus + ".png";
					document.getElementById("imageStim").src = serverRoot + stimPath + "statics/" + maintaskParam.trial.randStimulusFace + ".png";

					// if (maintaskParam.trial.stimulus[4] == 1) {
						document.getElementById("imageStim_front1").src = document.getElementById("imageStim").src;
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
					document.getElementById("miniface_This").src = document.getElementById("imageStim").src;

					var outcomeOther = 0;
					var outcomeThis = 0;
					if (maintaskParam.trial.decisionOther === "Split" && maintaskParam.trial.decisionThis === "Split") {
						outcomeOther = 'Won $' + numberWithCommas(Math.floor(maintaskParam.trial.pot * 50) / 100);
						outcomeThis = outcomeOther;

						document.getElementById("imageContext").src = serverRoot + "images/" + "CC.png";
						$('#context_outcome_front1').html(outcomeOther);
						$('#context_outcome_front2').html(outcomeOther);

					}
					if (maintaskParam.trial.decisionOther === "Split" && maintaskParam.trial.decisionThis === "Stole") {
						outcomeOther = 'Won $0.00';
						outcomeThis = 'Won $' + numberWithCommas(maintaskParam.trial.pot);

						// if (maintaskParam.trial.stimulus[4] == 1) {
							document.getElementById("imageContext").src = serverRoot + "images/" + "DC.png";
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
							document.getElementById("imageContext").src = serverRoot + "images/" + "CD.png";
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

						document.getElementById("imageContext").src = serverRoot + "images/" + "DD.png";
						$('#context_outcome_front1').html(outcomeOther);
						$('#context_outcome_front2').html(outcomeOther);
					}
					$('#context_outcomeOther').html(outcomeOther);
					$('#context_outcomeThis').html(outcomeThis);

					$("#contextTableFrontDiv").html('&nbsp;');
					$("#contextSubTableID").clone().appendTo("#contextTableFrontDiv"); // insert information in video div

					this.stimIDArray[maintaskParam.numComplete] = maintaskParam.trial.stimID;
					this.stimulusArray[maintaskParam.numComplete] = maintaskParam.trial.stimulus;
					this.randFaceId[maintaskParam.numComplete] = maintaskParam.trial.randStimulusFace;

					console.log(maintaskParam.numComplete + "  --------");
					console.log(maintaskParam.trial.stimulus);
					console.log("This: " + maintaskParam.trial.decisionThis);
					console.log("Other: " + maintaskParam.trial.decisionOther);


					maintaskParam.numComplete++;

					// Update progress bar
					$('.bar').css('width', Math.round(300.0 * maintaskParam.numComplete / maintaskParam.numTrials) + 'px');
					$("#trial-num").html(maintaskParam.numComplete);
					$("#total-num").html(maintaskParam.numTrials);
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
			}
		};

	});
}
